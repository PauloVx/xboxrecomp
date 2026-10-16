/*
 * ohci.c -- OHCI 1.0a host controller registers for the MCPX.
 *
 * See ohci.h for what this is and is not. The short version: enough for a
 * title's own USB driver to find a controller and a populated root hub port,
 * plus a trace of every register access, because what the driver does after
 * that decides how the rest gets built.
 */
#include "ohci.h"
#include "../platform/mmio_decode.h"
#include "../kernel/xbox_memory_layout.h"
#include "usb_gamepad.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* The runtime maps guest memory at a fixed host offset. */
extern ptrdiff_t xbox_GetMemoryOffset(void);

/* The controller's interrupt goes through the kernel's level-triggered line,
 * like every other device's: the timer thread runs the title's ISR while it
 * is up, under the IRQL model, and drains the DPC the ISR queues. */
extern void xbox_SetInterruptLine(uint32_t vector, int asserted);

/* Declared in the generated runtime; thread-local, so the values below are
 * this thread's and not the guest thread's. */
#if defined(_MSC_VER)
#  define OHCI_TLS __declspec(thread)
#else
#  define OHCI_TLS __thread
#endif
extern OHCI_TLS uint32_t g_eax, g_ecx, g_edx, g_esp;
extern OHCI_TLS uint32_t g_ebx, g_esi, g_edi;
extern OHCI_TLS uint32_t g_fs_base;
extern uint32_t xbox_AllocThreadTib(void);

/* The vector XPP takes for USB0. HalGetInterruptVector(1) returns 1 here, and
 * the bus interrupt level is what the XDK passes. */
#define OHCI_VECTOR  1

/* ---- OHCI 1.0a operational registers, by byte offset ------------------- */
#define HcRevision              0x00
#define HcControl               0x04
#define HcCommandStatus         0x08
#define HcInterruptStatus       0x0C
#define HcInterruptEnable       0x10
#define HcInterruptDisable      0x14
#define HcHCCA                  0x18
#define HcPeriodCurrentED       0x1C
#define HcControlHeadED         0x20
#define HcControlCurrentED      0x24
#define HcBulkHeadED            0x28
#define HcBulkCurrentED         0x2C
#define HcDoneHead              0x30
#define HcFmInterval            0x34
#define HcFmRemaining           0x38
#define HcFmNumber              0x3C
#define HcPeriodicStart         0x40
#define HcLSThreshold           0x44
#define HcRhDescriptorA         0x48
#define HcRhDescriptorB         0x4C
#define HcRhStatus              0x50
#define HcRhPortStatus1         0x54
/* Through port 4. The root hub here reports two downstream ports, and the
 * driver reads four status registers regardless -- so the file has to cover
 * them or the last one falls outside the array and is answered by the
 * not-a-register path instead of by an empty port. Same value either way,
 * reached honestly. */
#define OHCI_REG_MAX            0x64

/* HcCommandStatus */
#define CS_HCR                  0x00000001u   /* host controller reset       */

/* HcInterruptStatus / Enable */
#define INTR_SO                 0x00000001u   /* scheduling overrun          */
#define INTR_WDH                0x00000002u   /* writeback done head         */
#define INTR_SF                 0x00000004u   /* start of frame              */
#define INTR_RD                 0x00000008u   /* resume detected             */
#define INTR_UE                 0x00000010u   /* unrecoverable error         */
#define INTR_FNO                0x00000020u   /* frame number overflow       */
#define INTR_RHSC               0x00000040u   /* root hub status change      */

/* HccaFrameNumber is at +0x80 and the done head at +0x84. Declared up here
 * because the register write handler retires the done list on a WDH ack. */
#define HCCA_DONE_HEAD 0x84
static void wr32(uint32_t va, uint32_t v);
#define INTR_MIE                0x80000000u   /* master interrupt enable     */

/* HcRhPortStatus */
#define PORT_CCS                0x00000001u   /* current connect status      */
#define PORT_PES                0x00000002u   /* port enable status          */
#define PORT_PSS                0x00000004u   /* port suspend status         */
#define PORT_PPS                0x00000100u   /* port power status           */
#define PORT_LSDA               0x00000200u   /* low speed device attached   */
#define PORT_CSC                0x00010000u   /* connect status change       */
#define PORT_PESC               0x00020000u   /* enable status change        */
#define PORT_PRSC               0x00100000u   /* reset status change         */

/* Writes to HcRhPortStatus set/clear by bit position rather than by value. */
#define PORT_W_CCS_CLEAR_ENABLE 0x00000001u   /* ClearPortEnable             */
#define PORT_W_PES_SET_ENABLE   0x00000002u   /* SetPortEnable               */
#define PORT_W_PRS_SET_RESET    0x00000010u   /* SetPortReset                */
#define PORT_W_PPS_SET_POWER    0x00000100u   /* SetPortPower                */
#define PORT_W_CLEAR_POWER      0x00000200u   /* ClearPortPower              */

/* The first controller's root hub has the console's four front sockets.
 * XAPI numbers them so that root port 3 is controller 1, 4 is controller 2,
 * 1 is controller 3 and 2 is controller 4 -- the same mapping xemu wires its
 * pads to. A model with two ports could only ever present controller 3 or 4,
 * and BLACK, reading START from controller 3, never left its title screen. */
#define OHCI_PORTS              4

/* Root port (0-based) the pad appears on by default: controller 1. */
#define OHCI_PLAYER1_PORT       2

typedef struct {
    uint32_t base;                      /* Xbox VA of the register block   */
    uint32_t reg[OHCI_REG_MAX / 4];
    unsigned reads, writes, decode_fail;
    unsigned acks;                      /* status bits the driver cleared   */
    uint32_t done_queue;                /* retired TDs not yet written back */
    int      index;
} OhciController;

static OhciController *hc_for(uint32_t va);

static OhciController s_hc[2];

/* Pad reports served and when the last one was (GetTickCount64 ms), for the
 * hang report: whether the driver stopped polling or the list stopped being
 * walked is the first question when input dies. */
static volatile uint64_t s_reports, s_last_report_ms;
static int s_enabled;
static int s_trace;

static uint32_t *reg_of(OhciController *hc, uint32_t off)
{
    return (off < OHCI_REG_MAX) ? &hc->reg[off / 4] : NULL;
}

/* ---- register semantics ------------------------------------------------ */

static uint64_t ohci_read(void *dev, uint32_t off, int size)
{
    OhciController *hc = (OhciController *)dev;
    uint32_t aligned = off & ~3u;
    uint32_t *r = reg_of(hc, aligned);
    uint32_t v = r ? *r : 0;

    hc->reads++;

    /* HcFmRemaining and HcFmNumber advance on their own. A driver that waits
     * for the frame counter to move is waiting for the controller to be
     * running, and a counter that never changes is a controller that is not.
     * Derived from the read count rather than a timer: it only has to be
     * monotonic, and a timer here would need a thread to be worth having. */
    if (aligned == HcFmNumber)
        v = (hc->reads >> 3) & 0xFFFFu;
    else if (aligned == HcFmRemaining)
        v = (hc->reads * 977u) & 0x3FFFu;

    /* Sub-dword reads take their slice of the containing register. */
    if (size < 4) {
        unsigned shift = (off & 3u) * 8u;
        v >>= shift;
        if (size == 1) v &= 0xFFu;
        else if (size == 2) v &= 0xFFFFu;
    }

    if (s_trace) {
        static unsigned n;
        if (n++ < 600)
            fprintf(stderr, "  [OHCI%d] read  +0x%02X = %08X\n",
                    hc->index, off, (uint32_t)v);
    }
    return v;
}

static void ohci_write(void *dev, uint32_t off, uint64_t val, int size)
{
    OhciController *hc = (OhciController *)dev;
    uint32_t aligned = off & ~3u;
    uint32_t *r = reg_of(hc, aligned);
    uint32_t v = (uint32_t)val;

    hc->writes++;
    if (!r)
        return;

    if (s_trace) {
        static unsigned n;
        if (n++ < 600)
            fprintf(stderr, "  [OHCI%d] write +0x%02X = %08X\n",
                    hc->index, off, v);
    }

    switch (aligned) {
    case HcRevision:                    /* read-only */
        return;

    case HcCommandStatus:
        /* HCR is self-clearing: the controller resets and drops the bit, and
         * a driver polls for exactly that. Leaving it set is a hang, and it
         * is the first thing a driver does, so it would be the only thing
         * anyone ever saw of this file. */
        *r |= v;
        if (*r & CS_HCR) {
            *r &= ~CS_HCR;
            hc->reg[HcControl / 4] &= ~0xC0u;    /* back to UsbReset state  */
            hc->reg[HcInterruptStatus / 4] = 0;
            hc->reg[HcInterruptEnable / 4] = 0;
        }
        return;

    case HcInterruptStatus:
        /* Acknowledging WritebackDoneHead retires the done list.
         *
         * The controller publishes the queue in HccaDoneHead and raises WDH;
         * the driver reads the list and clears WDH to say it is finished, and
         * the head must then read as zero. Leaving the old value there is a
         * stale pointer to descriptors that have already been consumed, and a
         * driver that checks the head before trusting the interrupt walks
         * them a second time. That is a plausible way to enumerate a device
         * perfectly and then stop, which is what happened here. */
        if ((v & INTR_WDH) && (*r & INTR_WDH)) {
            uint32_t hcca = hc->reg[HcHCCA / 4];
            if (hcca)
                wr32(hcca + HCCA_DONE_HEAD, 0);
        }
        if (*r & v)
            hc->acks++;
        *r &= ~v;                       /* write 1 to clear                 */
        return;

    case HcInterruptEnable:
        hc->reg[HcInterruptEnable / 4] |= v;
        return;

    case HcInterruptDisable:
        hc->reg[HcInterruptEnable / 4] &= ~v;
        return;

    case HcFmNumber:
    case HcFmRemaining:
        return;                         /* driven by the controller         */

    case HcRhDescriptorA:
        /* NumberDownstreamPorts is ours; the driver may set the power and
         * over-current policy bits above it. */
        *r = (*r & 0x000000FFu) | (v & ~0x000000FFu);
        return;

    case HcRhPortStatus1:
    case HcRhPortStatus1 + 4:
    case HcRhPortStatus1 + 8:
    case HcRhPortStatus1 + 12: {
        /* Writes here are set/clear requests by bit position, not a value to
         * store. Getting that wrong looks like a port that will not enable. */
        unsigned port = (aligned - HcRhPortStatus1) / 4;
        uint32_t *ps = &hc->reg[(HcRhPortStatus1 + port * 4) / 4];

        /* Write-1-to-clear first, because the change bits a write carries
         * refer to the state BEFORE it.
         *
         * This ordering is the whole bug it once had. A driver starting a
         * fresh reset writes SetPortReset and ClearPortResetStatusChange in
         * the same word -- "acknowledge the last reset, begin another" --
         * which is both legal and what the Xbox USB stack does: DDS9 writes
         * 0x01100010. Clearing afterwards wiped the PRSC this very write had
         * just raised, so the reset never reported complete, the driver
         * reset the port again, and enumeration looped forever one step from
         * finishing. The port looked healthy the whole time: connected,
         * enabled, powered. */
        *ps &= ~(v & 0xFFFF0000u);

        if (v & PORT_W_CCS_CLEAR_ENABLE) *ps &= ~PORT_PES;
        if (v & PORT_W_PES_SET_ENABLE)   *ps |= (*ps & PORT_CCS) ? PORT_PES : 0;
        if (v & PORT_W_PPS_SET_POWER)    *ps |= PORT_PPS;
        if (v & PORT_W_CLEAR_POWER)      *ps &= ~PORT_PPS;
        if (v & PORT_W_PRS_SET_RESET) {
            /* Reset completes immediately -- there is no wire to settle -- so
             * PRS is never observed set. What matters is what the driver
             * checks afterwards: a present device comes back enabled, and the
             * reset-change bit says the reset finished.
             *
             * Only the status bit is set here. Delivering the interrupt is the
             * controller thread's job, because this runs on the guest's own
             * thread inside a fault handler, and pointing g_esp at a worker
             * stack from here would overwrite the stack pointer of the thread
             * being interrupted. */
            if (*ps & PORT_CCS)
                *ps |= PORT_PES;
            *ps |= PORT_PRSC;
            hc->reg[HcInterruptStatus / 4] |= INTR_RHSC;
        }
        return;
    }

    default:
        break;
    }

    if (size == 4) {
        *r = v;
    } else {
        unsigned shift = (off & 3u) * 8u;
        uint32_t mask = ((size == 1) ? 0xFFu : 0xFFFFu) << shift;
        *r = (*r & ~mask) | ((v << shift) & mask);
    }
}

/* ---- the control list -------------------------------------------------- */
/*
 * A host controller is a bus master: the driver builds endpoint and transfer
 * descriptors in RAM, points HcControlHeadED at them and sets ControlListFilled,
 * and the controller walks that list itself. Nothing arrives through MMIO, so
 * none of this is visible to the register trace -- which is why the driver
 * looked idle after the port came up.
 *
 * OHCI 1.0a, section 4. An endpoint descriptor is four dwords:
 *
 *   +0  FA | EN<<7 | D<<11 | S<<13 | K<<14 | F<<15 | MPS<<16
 *   +4  TailP        queue tail, 16-byte aligned
 *   +8  HeadP        queue head, with Halted in bit 0 and toggleCarry in bit 1
 *   +C  NextED
 *
 * and a general transfer descriptor is four more:
 *
 *   +0  ... DP<<19 | DI<<21 | T<<24 | EC<<26 | CC<<28
 *   +4  CBP          current buffer pointer, 0 when the transfer moved nothing
 *   +8  NextTD
 *   +C  BE           last byte of the buffer, inclusive
 *
 * A transfer is done when HeadP reaches TailP. Completed descriptors go on the
 * done queue, newest first, and the controller publishes it in the HCCA and
 * raises WritebackDoneHead.
 */
#define ED_SKIP        (1u << 14)
#define ED_HEAD_HALT   (1u << 0)
#define ED_HEAD_TOGGLE (1u << 1)
#define ED_PTR_MASK    0xFFFFFFF0u

#define TD_DP_SETUP    0u
#define TD_DP_OUT      1u
#define TD_DP_IN       2u
#define TD_CC_NOERROR  0u
#define TD_CC_STALL    4u


/* Each pad on the bus has its own control pipe. */
static uint32_t g_setup_pending_[USB_GAMEPAD_MAX];   /* a SETUP awaits its data stage */
static UsbSetup g_setup_[USB_GAMEPAD_MAX];
#define g_setup_pending g_setup_pending_[pad]
#define g_setup         g_setup_[pad]

/* The control transfer's data stage, across however many descriptors it takes.
 *
 * bMaxPacketSize0 is 8, so a driver asking for a 32-byte configuration
 * descriptor queues four 8-byte IN descriptors and expects successive chunks.
 * Answering each one from the top of the buffer hands it the first 8 bytes
 * four times: the descriptor it assembles is nonsense, it gives up and starts
 * again, which is exactly the loop DDS9 sat in -- GET_DESCRIPTOR, SET_ADDRESS,
 * GET_DESCRIPTOR, forever. The answer is computed once per setup packet and
 * then consumed. */
static uint8_t g_ctrl_buf_[USB_GAMEPAD_MAX][64];
static int     g_ctrl_len_[USB_GAMEPAD_MAX] = { -1, -1 };   /* -1 = not answered yet */
static int     g_ctrl_sent_[USB_GAMEPAD_MAX];
#define g_ctrl_buf  g_ctrl_buf_[pad]
#define g_ctrl_len  g_ctrl_len_[pad]
#define g_ctrl_sent g_ctrl_sent_[pad]

/* Pads plugged in so far, in port order (RECOMP_USB_PADS, default 2: the second is split-screen's player two). */
static int s_npads = 2, s_plugged_pads;

/* The pad an endpoint descriptor talks to: by its function address; address 0, the default one a device answers
 * before SET_ADDRESS, is the plugged pad not given an address yet (pads are plugged one at a time). */
static int ohci_pad_for(uint32_t ed0)
{
    const uint32_t fa = ed0 & 0x7Fu;
    int d;
    for (d = 0; d < s_plugged_pads; d++)
        if (fa ? usb_gamepad_address(d) == fa : usb_gamepad_address(d) == 0)
            return d;
    return -1;
}

/* Every address below came out of guest memory, so none of them are trusted.
 *
 * This is not theoretical. The driver writes HcControlHeadED twice during
 * bring-up, and the second write is 0xCCCCCCCC -- MSVC's uninitialised-memory
 * fill, left there by a local that was never assigned. Following it lands far
 * outside the mapping and takes the runtime down with an access violation,
 * which is a crash the title itself would never have had. A device model
 * following a pointer a title left lying around has to check it first.
 */
/* The contiguous (physical-mirror) window, where physical memory is. */
#define OHCI_CONTIG_BASE 0x80000000u
#define OHCI_CONTIG_SIZE (64u * 1024u * 1024u)
/* Physical addresses decode through 128 MB: the window, then low RAM's own
 * view above it (xbox_PhysicalAddressOf), both at 0x80000000 + P. */
#define OHCI_PHYS_SIZE   (2u * OHCI_CONTIG_SIZE)

/* A DMA address as the controller sees it, as the guest VA holding it.
 *
 * The registers and descriptors hold physical addresses: XAPI fills them from
 * MmGetPhysicalAddress, which answers from the contiguous window, so physical
 * P is window VA 0x80000000 + P -- the same rule the GPU and APU models use.
 * A driver that hands over the window's VA instead means the same page (the
 * hardware decodes only the low bits), and DDS9's does: its first control
 * transfer sits at 0x80408940. Anything else -- 0xCCCCCCCC, MSVC's fill for
 * an unassigned local, which the driver writes to HcControlHeadED during
 * bring-up -- is not memory and maps to 0.
 *
 * Reading physical addresses as low guest addresses, as this used to, walked
 * BLACK's descriptors at 0x033074C0 in the wrong storage: every one read as
 * zero, so enumeration never moved. */
static uint32_t dma_va(uint32_t addr)
{
    if (addr >= OHCI_CONTIG_BASE && addr - OHCI_CONTIG_BASE < OHCI_PHYS_SIZE)
        return addr;
    if (addr && addr < OHCI_PHYS_SIZE)
        return OHCI_CONTIG_BASE + addr;
    return 0;
}

static int guest_ok(uint32_t addr, uint32_t bytes)
{
    uint32_t va = dma_va(addr);

    if (va == 0)
        return 0;
    return (uint64_t)va + bytes <= (uint64_t)OHCI_CONTIG_BASE + OHCI_PHYS_SIZE;
}

static uint32_t rd32(uint32_t addr)
{
    if (!guest_ok(addr, 4))
        return 0;
    return *(uint32_t *)((uint8_t *)xbox_GetMemoryOffset() + dma_va(addr));
}
static void wr32(uint32_t addr, uint32_t v)
{
    if (!guest_ok(addr, 4))
        return;
    *(uint32_t *)((uint8_t *)xbox_GetMemoryOffset() + dma_va(addr)) = v;
}
static uint8_t *guest_ptr(uint32_t addr, uint32_t bytes)
{
    return guest_ok(addr, bytes)
         ? (uint8_t *)xbox_GetMemoryOffset() + dma_va(addr) : NULL;
}

/* Move one transfer descriptor. Returns the condition code to report. */
static uint32_t ohci_do_td(OhciController *hc, uint32_t ed0, uint32_t td)
{
    uint32_t info = rd32(td);
    uint32_t cbp  = rd32(td + 4);
    uint32_t be   = rd32(td + 12);
    uint32_t dp   = (info >> 19) & 3u;
    int      len  = (cbp && be >= cbp) ? (int)(be - cbp + 1) : 0;
    uint32_t endpoint = (ed0 >> 7) & 0xFu;
    int      moved = 0;
    const int pad = ohci_pad_for(ed0);

    if (pad < 0)
        return TD_CC_NOERROR;   /* no device at that address */

    if (s_trace) {
        static unsigned n;
        if (n++ < 80) {
            fprintf(stderr, "  [OHCI%d] TD %08X dp=%u ep=%u cbp=%08X be=%08X "
                            "len=%d\n", hc->index, td, dp, endpoint, cbp, be,
                    len);
            fflush(stderr);
        }
    }
    if (dp == TD_DP_SETUP) {
        /* Eight bytes of setup, kept for the data stage that follows. */
        if (len >= 8) {
            const uint8_t *p = guest_ptr(cbp, 8);
            if (!p) return TD_CC_NOERROR;
            g_setup.bmRequestType = p[0];
            g_setup.bRequest      = p[1];
            g_setup.wValue        = (uint16_t)(p[2] | (p[3] << 8));
            g_setup.wIndex        = (uint16_t)(p[4] | (p[5] << 8));
            g_setup.wLength       = (uint16_t)(p[6] | (p[7] << 8));
            g_setup_pending = 1;
            g_ctrl_len = -1;   /* answered lazily on the first IN */
            g_ctrl_sent = 0;
            if (s_trace) {
                fprintf(stderr, "  [OHCI%d] SETUP %02X %02X value %04X "
                                "index %04X len %u\n",
                        hc->index, g_setup.bmRequestType, g_setup.bRequest,
                        g_setup.wValue, g_setup.wIndex, g_setup.wLength);
                fflush(stderr);
            }
            moved = 8;
        }
    } else if (dp == TD_DP_IN) {
        if (endpoint == 0) {
            /* Data stage of a control transfer, or its status stage when the
             * driver asks for nothing. */
            int n;

            if (!g_setup_pending)
                return TD_CC_NOERROR;
            if (g_ctrl_len < 0) {
                g_ctrl_len = usb_gamepad_control(pad, &g_setup, g_ctrl_buf,
                                                 (int)sizeof g_ctrl_buf);
                if (g_ctrl_len < 0) {
                    g_setup_pending = 0;
                    return TD_CC_STALL;
                }
                g_ctrl_sent = 0;
            }
            /* Whatever is left, capped by this descriptor's buffer. A short
             * packet is how the device says "that is all", so running out is
             * the normal end of the stage rather than an error. */
            n = g_ctrl_len - g_ctrl_sent;
            if (n > len) n = len;
            if (n < 0) n = 0;
            if (n > 0 && guest_ptr(cbp, (uint32_t)n))
                memcpy(guest_ptr(cbp, (uint32_t)n),
                       g_ctrl_buf + g_ctrl_sent, (size_t)n);
            g_ctrl_sent += n;
            moved = n;
            if (s_trace && n > 0) {
                fprintf(stderr, "  [OHCI%d] IN ep0 %d bytes (%d/%d)\n",
                        hc->index, n, g_ctrl_sent, g_ctrl_len);
                fflush(stderr);
            }
        } else {
            /* The pad's report, on its interrupt endpoint. */
            uint8_t rep[32];
            int n = usb_gamepad_report(pad, rep, (int)sizeof rep);
            s_reports++;
            s_last_report_ms = GetTickCount64();
            if (n > len) n = len;
            if (n > 0 && guest_ptr(cbp, (uint32_t)n))
                memcpy(guest_ptr(cbp, (uint32_t)n), rep, (size_t)n);
            moved = n;
        }
    } else {
        /* OUT: the status stage of an IN control transfer, or rumble. Both
         * are accepted and discarded. */
        moved = len;
        g_setup_pending = 0;
    }

    /* CBP is zero when everything asked for moved, and otherwise points past
     * what did. A driver computes the transferred length from it. */
    wr32(td + 4, (moved >= len) ? 0u : cbp + (uint32_t)moved);
    return TD_CC_NOERROR;
}

/* Walk the control list once. Returns 1 if anything completed. */
/* Walk one list of endpoint descriptors, moving whatever is queued on each.
 *
 * Shared, because the control list is not the only one that matters: a gamepad
 * delivers its reports on an interrupt endpoint, and those EDs hang off the
 * HCCA interrupt table rather than HcControlHeadED. Walking only the control
 * list enumerates a device perfectly and then never hears from it again. */
static int ohci_walk_eds(OhciController *hc, uint32_t ed, uint32_t *done_head)
{
    int completed = 0, guard = 0;

    while (guest_ok(ed, 16) && ++guard < 64) {
        uint32_t ed0  = rd32(ed);
        uint32_t tail = rd32(ed + 4) & ED_PTR_MASK;
        uint32_t head = rd32(ed + 8);
        int tguard = 0;

        if (s_trace) {
            static unsigned n;
            if (n++ < 60) {
                fprintf(stderr, "  [OHCI%d] ED %08X info=%08X head=%08X "
                                "tail=%08X next=%08X\n", hc->index, ed, ed0,
                        head, tail, rd32(ed + 12));
                fflush(stderr);
            }
        }
        if (ed0 & ED_SKIP) { ed = rd32(ed + 12) & ED_PTR_MASK; continue; }
        if (head & ED_HEAD_HALT) { ed = rd32(ed + 12) & ED_PTR_MASK; continue; }

        while ((head & ED_PTR_MASK) != tail
            && guest_ok(head & ED_PTR_MASK, 16) && ++tguard < 64) {
            uint32_t td = head & ED_PTR_MASK;
            uint32_t next = rd32(td + 8) & ED_PTR_MASK;
            uint32_t cc = ohci_do_td(hc, ed0, td);

            /* Report the outcome where the driver reads it, then put the
             * descriptor on the done queue, newest first. */
            wr32(td, (rd32(td) & 0x0FFFFFFFu) | (cc << 28));
            wr32(td + 8, *done_head);
            *done_head = td;
            completed++;

            head = next | (head & ED_HEAD_TOGGLE);
            if (cc != TD_CC_NOERROR) {
                head |= ED_HEAD_HALT;       /* a stall halts the endpoint */
                break;
            }
        }
        wr32(ed + 8, head);
        ed = rd32(ed + 12) & ED_PTR_MASK;
    }

    return completed;
}

/* Write the done queue back, as a controller does at the end of a frame --
 * but only once the driver has taken the last one (WDH clear). Until then
 * retired descriptors wait in the controller's own queue.
 *
 * It used to write every list's completions straight into HccaDoneHead. The
 * control, bulk and periodic lists each did so in the same tick, so a later
 * list's queue replaced an earlier one's, and a new tick replaced a queue the
 * driver had not read yet. A lost completion is a transfer the driver never
 * sees finish, and a lost interrupt-IN completion is a pad nobody polls again:
 * BLACK's input stopped for good soon after its title screen. */
static void ohci_publish_done(OhciController *hc)
{
    uint32_t hcca = hc->reg[HcHCCA / 4];

    if (!hc->done_queue || (hc->reg[HcInterruptStatus / 4] & INTR_WDH))
        return;
    if (hcca)
        wr32(hcca + HCCA_DONE_HEAD, hc->done_queue);
    hc->done_queue = 0;
    hc->reg[HcDoneHead / 4] = 0;
    hc->reg[HcInterruptStatus / 4] |= INTR_WDH;
}

static int ohci_run_control_list(OhciController *hc)
{
    return ohci_walk_eds(hc, hc->reg[HcControlHeadED / 4] & ED_PTR_MASK,
                         &hc->done_queue);
}

/* The periodic list for the current frame.
 *
 * The HCCA opens with 32 ED pointers, one per frame slot, and the controller
 * services the one the frame number selects. An interrupt endpoint polled
 * every 4 ms appears in several slots, so cycling through them the way the
 * frame counter does is what makes its transfers happen at all. */
static int ohci_run_periodic_list(OhciController *hc)
{
    uint32_t hcca = hc->reg[HcHCCA / 4];
    uint32_t slot, ed;
    int completed;

    if (!hcca)
        return 0;
    /* Every slot, not the one the frame counter happens to land on.
     *
     * A tick covers twenty frames, and stepping the counter by twenty means
     * only slots that are multiples of four are ever selected -- gcd(20,32)
     * is 4, so twenty-four of the thirty-two are never looked at. An
     * interrupt endpoint that lands in one of those is polled by the driver
     * and serviced by nobody. Sweeping all of them costs a guest read per
     * slot and cannot miss. */
    completed = 0;
    for (slot = 0; slot < 32u; slot++) {
        ed = rd32(hcca + slot * 4u) & ED_PTR_MASK;
        if (ed)
            completed += ohci_walk_eds(hc, ed, &hc->done_queue);
    }
    return completed;
}

/* ---- raising an interrupt --------------------------------------------- */

/* The line, as OHCI defines it: asserted while MIE is on and an enabled
 * source is set. Recomputed on every register access and every tick.
 *
 * It used to call the ISR itself, on this controller's thread. That ran the
 * title's USB interrupt code outside the IRQL model -- concurrently with any
 * thread the driver thought it had locked out by raising IRQL -- and its DPC
 * then ran on the timer thread with nothing ordering the two. Only HC0 has a
 * device here, and OHCI_VECTOR is HC0's. */
static void ohci_update_line(OhciController *hc)
{
    uint32_t enable = hc->reg[HcInterruptEnable / 4];
    uint32_t pending = hc->reg[HcInterruptStatus / 4] & enable & 0x7Fu;

    if (hc->index != 0)
        return;
    xbox_SetInterruptLine(OHCI_VECTOR, (enable & INTR_MIE) && pending);
}

/* ---- bring-up ---------------------------------------------------------- */

static void ohci_reset(OhciController *hc, uint32_t base, int index)
{
    memset(hc, 0, sizeof *hc);
    hc->base  = base;
    hc->index = index;

    hc->reg[HcRevision / 4]       = 0x00000010u;   /* OHCI 1.0             */
    hc->reg[HcFmInterval / 4]     = 0x27782EDFu;   /* 11999, FSMPS default */
    hc->reg[HcPeriodicStart / 4]  = 0x00003E67u;   /* 90% of the frame     */
    hc->reg[HcLSThreshold / 4]    = 0x00000628u;

    /* Root hub: OHCI_PORTS downstream, ports always powered, no over-current
     * reporting. NoPowerSwitching keeps a driver from waiting on a power-on
     * sequence that has nothing to switch. */
    hc->reg[HcRhDescriptorA / 4]  = (uint32_t)OHCI_PORTS | (1u << 9);
    hc->reg[HcRhDescriptorB / 4]  = 0x00000000u;
    hc->reg[HcRhStatus / 4]       = 0x00000000u;

    /* Ports powered and empty. The gamepad is not here yet, deliberately.
     *
     * Presenting it as already connected does not work, and the reason is
     * worth keeping: the driver scans the root hub itself during bring-up,
     * sees the connect-status-change bit, clears it, and by the time it
     * unmasks the root hub interrupt there is no change left to report. The
     * pending status bit does not survive either, because the driver resets
     * the controller first and a reset clears interrupt status -- both of
     * those are correct behaviour, and between them a device that was always
     * there is a device that never arrives.
     *
     * A console detects the port change after the controller is running, so
     * that is what the controller thread does: it waits for operational with
     * the interrupt unmasked, and only then plugs the device in. */
    {
        unsigned port;
        for (port = 0; port < OHCI_PORTS; port++)
            hc->reg[(HcRhPortStatus1 + port * 4) / 4] = PORT_PPS;
    }
}

/* The controller, running on its own thread.
 *
 * It has to be its own thread for a reason that is easy to get wrong: the
 * guest register file is thread-local, so setting g_esp to a worker stack on
 * the guest's thread would overwrite the guest's own stack pointer mid-call.
 * A separate thread has its own copy, and it also happens to be what the
 * hardware does -- an interrupt arrives when the controller decides, not when
 * the driver next reads a register.
 *
 * ponytail: one delivery, of the root hub status change that is already
 * pending from bring-up. There is nothing behind it yet -- no descriptor list
 * walking, so an enumeration attempt has nothing to answer it -- and the point
 * of this delivery is to find out what the driver does when it finally gets
 * the interrupt it has been waiting for. Repeat delivery and the transfer
 * lists come after that answer, not before it.
 */
static DWORD WINAPI ohci_thread(LPVOID unused)
{
    /* This thread calls recompiled code, so it needs what any thread running
     * recompiled code needs: its own TIB. The guest register set is already
     * thread-local and the ISR gets a worker stack, but fs:[0] is the SEH
     * chain head and fs:[4] reaches the CRT's per-thread data -- and a guest
     * function with an SEH prologue on a thread whose g_fs_base is zero
     * dereferences null before it executes a line of its own body. That is
     * what killed the process here, two interrupts in, with no fault report
     * because the fault was in the runtime rather than in the title. */
    unsigned waited = 0;
    int plugged = 0;
    unsigned first_port = 0, settle = 0;
    uint32_t last_status = 0;
    unsigned repeats = 0, last_acks = 0;

    (void)unused;
    {
        uint32_t tib = xbox_AllocThreadTib();
        if (!tib) {
            fprintf(stderr, "  [OHCI0] no TIB for the controller thread; "
                            "not delivering interrupts\n");
            fflush(stderr);
            return 0;
        }
        g_fs_base = tib;
    }

    for (;;) {
        OhciController *hc = &s_hc[0];
        uint32_t control, enable, status;

        Sleep(20);
        control = hc->reg[HcControl / 4];
        enable  = hc->reg[HcInterruptEnable / 4];

        /* Operational is HCFS == 10b in bits 7:6. Interrupting a controller
         * the driver has not started yet is not a test of anything. */
        if ((control & 0xC0u) != 0x80u) {
            if (++waited > 1500)              /* 30 s and it never started */
                break;
            continue;
        }
        if (!(enable & INTR_MIE))
            continue;

        /* Plug the device in once, after the driver is running and listening.
         * Presenting it earlier does not work: the driver clears the connect
         * change during its own bring-up scan, and a reset clears interrupt
         * status, so a device that was always there is one that never
         * arrives. */
        if (!plugged && (enable & INTR_RHSC)) {
            /* Which root port the device appears on decides which controller
             * XAPI makes it (see OHCI_PORTS). Controller 1 by default;
             * RECOMP_USB_PORT picks another root port, 0..3. */
            const char *pspec = getenv("RECOMP_USB_PORT");
            unsigned port = pspec ? (unsigned)atoi(pspec) : OHCI_PLAYER1_PORT;
            if (port >= OHCI_PORTS) port = OHCI_PLAYER1_PORT;
            hc->reg[(HcRhPortStatus1 + port * 4) / 4] |= PORT_CCS | PORT_CSC;
            hc->reg[HcInterruptStatus / 4] |= INTR_RHSC;
            plugged = 1;
            s_plugged_pads = 1;
            first_port = port;
            fprintf(stderr, "  [OHCI0] operational after %u ms; device "
                            "arriving on root port %u\n", waited * 20,
                    port + 1);
            fflush(stderr);
        }

        /* The second pad arrives about a second after the first is configured, on the next root port, so the
         * driver enumerates them in turn and only one device is ever at the default address. */
        if (plugged && s_plugged_pads < s_npads && usb_gamepad_configured(s_plugged_pads - 1) && ++settle > 50) {
            const unsigned port = (first_port + (unsigned)s_plugged_pads) % OHCI_PORTS;
            hc->reg[(HcRhPortStatus1 + port * 4) / 4] |= PORT_CCS | PORT_CSC;
            hc->reg[HcInterruptStatus / 4] |= INTR_RHSC;
            s_plugged_pads++;
            settle = 0;
            fprintf(stderr, "  [OHCI0] pad %d arriving on root port %u\n", s_plugged_pads, port + 1);
            fflush(stderr);
        }

        /* The frame clock.
         *
         * INTR_SF was defined and never raised, and HcFmNumber only moved when
         * something read it. That is enough for a driver that polls, and not
         * enough for one that waits: USB enumeration is full of delays counted
         * in frames -- the recovery time after SET_ADDRESS is the first -- and
         * the Xbox stack enables StartOfFrame precisely to count them. Without
         * a frame ever starting, enumeration stops one transfer past
         * SET_ADDRESS with the driver waiting for time to pass.
         *
         * Twenty frames a tick because a frame is 1 ms and this loop sleeps
         * 20. The count is what drivers compare; the rate is what they feel,
         * and both are closer to right this way than a counter that never
         * moves. */
        {
            uint32_t hcca = hc->reg[HcHCCA / 4];

            hc->reg[HcFmNumber / 4] =
                (hc->reg[HcFmNumber / 4] + 20u) & 0xFFFFu;
            /* HccaFrameNumber is 16 bits at +0x80 with a pad above it that the
             * controller zeroes, so a dword write is what hardware does. */
            if (hcca)
                wr32(hcca + 0x80, hc->reg[HcFmNumber / 4]);
            hc->reg[HcInterruptStatus / 4] |= INTR_SF;
        }

        /* Be the bus master. ControlListEnable in HcControl says the driver
         * wants the list walked; ControlListFilled says it has put something
         * on it. Walking on the tick rather than only when CLF is written
         * costs a read of a guest dword and means a descriptor queued without
         * rewriting CLF is still moved -- which is legal, and drivers do it. */
        /* PeriodicListEnable, bit 2 of HcControl. */
        if (control & 0x04u)
            ohci_run_periodic_list(hc);

        /* BulkListEnable, bit 5. Nothing on this device uses bulk, but the
         * list is walked the same way and a driver that puts a transfer
         * there is owed the same service. */
        if ((control & 0x20u)
         && guest_ok(hc->reg[HcBulkHeadED / 4] & ED_PTR_MASK, 16)) {
            if (ohci_walk_eds(hc, hc->reg[HcBulkHeadED / 4] & ED_PTR_MASK,
                              &hc->done_queue))
                hc->reg[HcCommandStatus / 4] &= ~0x04u;   /* BLF consumed */
        }

        if ((control & 0x10u)
         && guest_ok(hc->reg[HcControlHeadED / 4] & ED_PTR_MASK, 16)) {
            if (ohci_run_control_list(hc))
                hc->reg[HcCommandStatus / 4] &= ~0x02u;   /* CLF consumed */
        }

        /* End of the frame: hand the driver whatever retired, if it has
         * taken the last batch. */
        ohci_publish_done(hc);

        /* Level-triggered, which is what OHCI is: while an enabled source is
         * set, the line is asserted, and the kernel keeps running the ISR. */
        ohci_update_line(hc);
        status = hc->reg[HcInterruptStatus / 4] & enable & 0x7Fu;
        if (!status)
            continue;

        /* A stuck source is one the driver never acknowledges. StartOfFrame
         * is raised afresh every tick, so the same status on every tick is
         * normal; the same status with no status bit cleared in between is
         * not. Said once, and the controller keeps running. */
        if (status == last_status && hc->acks == last_acks) {
            if (++repeats == 200) {
                fprintf(stderr, "  [OHCI0] status %08X pending for 200 ticks "
                                "with no acknowledgement\n", status);
                fflush(stderr);
            }
        } else {
            last_status = status;
            last_acks = hc->acks;
            repeats = 0;
        }
    }
    return 0;
}

#if defined(_WIN32)
/* The controllers' own fault handler, so a title needs no wiring of its own:
 * an access to either trapped block is serviced here, anything else is passed
 * on. First in the chain, like the other register models. */
static LONG CALLBACK ohci_veh(PEXCEPTION_POINTERS ep)
{
    uintptr_t fault;
    uint32_t va;

    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        ep->ExceptionRecord->NumberParameters < 2)
        return EXCEPTION_CONTINUE_SEARCH;
    fault = (uintptr_t)ep->ExceptionRecord->ExceptionInformation[1];
    va = (uint32_t)(fault - (uintptr_t)xbox_GetMemoryOffset());
    if (!hc_for(va))
        return EXCEPTION_CONTINUE_SEARCH;
    return xbox_OhciHandleMmio(ep->ContextRecord, va)
               ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
#endif

void xbox_OhciInit(void)
{
    static int done;

    if (done)
        return;
    done = 1;

    /* Opt-in. Without a descriptor list walker behind it a driver that finds
     * a port has nothing to enumerate, so this must not change how a title
     * behaves until the rest of it exists. */
    if (!getenv("RECOMP_USB"))
        return;

    s_enabled = 1;
    s_trace   = getenv("RECOMP_USB_TRACE") != NULL;
    {
        const char *pads = getenv("RECOMP_USB_PADS");
        if (pads && *pads >= '1' && *pads <= '0' + USB_GAMEPAD_MAX) s_npads = *pads - '0';
    }
    ohci_reset(&s_hc[0], XBOX_OHCI0_BASE, 0);
    ohci_reset(&s_hc[1], XBOX_OHCI1_BASE, 1);

#if defined(_WIN32)
    /* The registers have to fault to be answered. The MCPX aperture is mapped
     * as plain committed memory, so both blocks are made inaccessible here and
     * the title's VEH routes the faults back to xbox_OhciHandleMmio.
     *
     * A failed protect switches the model off rather than leaving it half on:
     * a controller whose registers read as zero out of RAM is exactly the
     * situation this exists to end, and it would look identical. */
    {
        ptrdiff_t off = xbox_GetMemoryOffset();
        int i;

        if (!off) {
            s_enabled = 0;
            fprintf(stderr, "  OHCI: guest memory not mapped yet; disabled\n");
            return;
        }
        if (!AddVectoredExceptionHandler(1, ohci_veh)) {
            s_enabled = 0;
            fprintf(stderr, "  OHCI: no fault handler; disabled\n");
            return;
        }
        xbox_AddHangReport(xbox_OhciReport);
        for (i = 0; i < 2; i++) {
            DWORD old_protect;
            LPVOID at = (LPVOID)((uintptr_t)off + s_hc[i].base);
            if (!VirtualProtect(at, XBOX_OHCI_SIZE, PAGE_NOACCESS,
                                &old_protect)) {
                s_enabled = 0;
                fprintf(stderr, "  OHCI: cannot trap 0x%08X (error %lu); "
                                "disabled\n", s_hc[i].base, GetLastError());
                return;
            }
        }
    }
#endif

    fprintf(stderr, "  OHCI: two controllers at 0x%08X and 0x%08X, "
                    "%d ports each, one pad on HC0 (controller 1 unless "
                    "RECOMP_USB_PORT picks another root port)\n",
            XBOX_OHCI0_BASE, XBOX_OHCI1_BASE, OHCI_PORTS);
    fflush(stderr);

#if defined(_WIN32)
    {
        HANDLE th = CreateThread(NULL, 0, ohci_thread, NULL, 0, NULL);
        if (th)
            CloseHandle(th);
    }
#endif
}

static OhciController *hc_for(uint32_t va)
{
    int i;

    if (!s_enabled)
        return NULL;
    for (i = 0; i < 2; i++)
        if (va >= s_hc[i].base && va < s_hc[i].base + XBOX_OHCI_SIZE)
            return &s_hc[i];
    return NULL;
}

int xbox_OhciOwnsAddress(uint32_t xbox_va)
{
    return hc_for(xbox_va) != NULL;
}

int xbox_OhciHandleMmio(void *ctx, uint32_t xbox_va)
{
#if defined(_WIN32)
    OhciController *hc = hc_for(xbox_va);
    int ok;

    if (!hc)
        return 0;
    ok = mmio_emulate((PCONTEXT)ctx, xbox_va - hc->base, hc,
                      ohci_read, ohci_write);
    if (ok)
        ohci_update_line(hc);       /* an ack or a mask change moves it */
    if (!ok && hc->decode_fail++ < 20) {
        const uint8_t *ip = (const uint8_t *)((PCONTEXT)ctx)->Rip;
        fprintf(stderr, "  [OHCI%d] undecoded access at +0x%03X: "
                        "%02X %02X %02X %02X %02X %02X\n",
                hc->index, xbox_va - hc->base,
                ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]);
        fflush(stderr);
    }
    return ok;
#else
    (void)ctx; (void)xbox_va;
    return 0;
#endif
}

void xbox_OhciReport(void)
{
    int i;

    if (!s_enabled)
        return;
    for (i = 0; i < 2; i++)
        fprintf(stderr, "  [OHCI%d] %u reads, %u writes, %u undecoded; "
                        "HcControl=%08X HcIntEnable=%08X HcIntStatus=%08X "
                        "HCCA=%08X port3=%08X\n",
                i, s_hc[i].reads, s_hc[i].writes, s_hc[i].decode_fail,
                s_hc[i].reg[HcControl / 4],
                s_hc[i].reg[HcInterruptEnable / 4],
                s_hc[i].reg[HcInterruptStatus / 4],
                s_hc[i].reg[HcHCCA / 4],
                s_hc[i].reg[(HcRhPortStatus1 + 8) / 4]);
    fprintf(stderr, "  [OHCI] %llu pad reports served, the last %llu ms ago\n",
            (unsigned long long)s_reports,
            s_last_report_ms ? (unsigned long long)(GetTickCount64()
                                                   - s_last_report_ms) : 0ull);
    fflush(stderr);
}
