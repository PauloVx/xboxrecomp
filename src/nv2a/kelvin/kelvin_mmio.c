/*
 * Kelvin: the NV2A registers that behave rather than store.
 *
 * The GPU's interrupt protocol lives in three register blocks: PMC, whose
 * INTR_0 summarises which unit is interrupting; PGRAPH, with the graphics
 * engine's status, the notify trap and the flip read index; and PCRTC, with
 * vblank. Their status registers are write-1-to-clear, PMC_INTR_0 is computed
 * from the others, and a few PGRAPH registers are triggers. None of that works
 * as plain memory -- an ISR acknowledging an interrupt writes the pending bit
 * back rather than clearing it -- so while the engine runs these pages are
 * PAGE_NOACCESS and every access by the title is serviced here, decoded by
 * platform/mmio_decode.h.
 *
 * Semantics follow xemu: pmc.c, pcrtc.c, pgraph_read/pgraph_write and
 * nv2a_update_irq. A register without behaviour keeps plain storage, so a
 * value the title wrote reads back as before; the one departure from xemu is
 * that unlisted PMC and PCRTC registers store rather than read as zero.
 *
 * The interrupt is level-triggered, as on hardware: the line is up while an
 * enabled source is pending and PMC_INTR_EN_0 is set. The kernel's interrupt
 * thread polls kelvin_irq_asserted() and runs the title's ISR; this module
 * signals kelvin_irq_event() so that thread need not wait out its tick.
 * Not modelled: PFIFO, PBUS, PTIMER and PVIDEO interrupts (nothing here
 * raises them), and hardware context switching.
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2015 Jannik Vogel
 * Copyright (c) 2018-2025 Matt Borgerson
 * (xemu PMC/PCRTC/PGRAPH register semantics this follows)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <windows.h>

#include "kelvin_int.h"
#include "kelvin.h"
#include "platform/mmio_decode.h"
#include "../nv2a_dac.h"

#define PMC_BASE     0x000000u
#define PMC_SIZE     0x1000u
#define PGRAPH_SIZE  0x2000u
#define PCRTC_BASE   0x600000u
#define PCRTC_SIZE   0x1000u
#define PRMDIO_BASE  0x681000u
#define PRMDIO_SIZE  0x1000u

static uint32_t s_pmc[PMC_SIZE / 4];
static uint32_t s_pgraph[PGRAPH_SIZE / 4];
static uint32_t s_pcrtc[PCRTC_SIZE / 4];
static uint32_t s_prmdio[PRMDIO_SIZE / 4];
static uint32_t s_pcrtc_raster;

static KelvinState *s_pg;
static SRWLOCK s_lock = SRWLOCK_INIT;
static volatile LONG s_waiters;
static HANDLE s_irq_event;
static PVOID s_veh;
static uint8_t *s_ap;             /* host address of aperture offset 0 */
static volatile LONG s_trapped;

static uint64_t s_irqs_raised;

void kelvin_lock(void)
{
    AcquireSRWLockExclusive(&s_lock);
}

void kelvin_unlock(void)
{
    ReleaseSRWLockExclusive(&s_lock);
}

bool kelvin_lock_wanted(void)
{
#ifdef _MSC_VER
    return ReadAcquire(&s_waiters) != 0;
#else
    return __atomic_load_n(&s_waiters, __ATOMIC_ACQUIRE) != 0;
#endif
}

/* The engine lets the lock go as soon as a register access is waiting (see
 * pusher_run). SRW locks are not fair, so after each release it steps aside
 * when one is. */
void kelvin_lock_yield(void)
{
    if (InterlockedCompareExchange(&s_waiters, 0, 0)) {
        SwitchToThread();
    }
}

/* ---- the interrupt line ------------------------------------------------------ */

/* NV_PMC_INTR_0: one bit per unit with an enabled interrupt pending. Lock
 * held. */
static uint32_t pmc_pending(void)
{
    uint32_t p = 0;

    if (s_pcrtc[NV_PCRTC_INTR_0 / 4] & s_pcrtc[NV_PCRTC_INTR_EN_0 / 4]) {
        p |= NV_PMC_INTR_0_PCRTC;
    }
    if (s_pg->pending_interrupts & s_pg->enabled_interrupts) {
        p |= NV_PMC_INTR_0_PGRAPH;
    }
    return p;
}

static bool line_up(void)
{
    return pmc_pending() && s_pmc[NV_PMC_INTR_EN_0 / 4];
}

void kelvin_irq_update(void)
{
    if (line_up()) {
        s_irqs_raised++;
        if (s_irq_event) {
            SetEvent(s_irq_event);
        }
    }
}

int kelvin_irq_asserted(void)
{
    int up;

    if (!InterlockedCompareExchange(&s_trapped, 0, 0)) {
        return 0;
    }
    /* The timer polls this immediately after raising vblank. Announce the
     * wait just like vblank and guest register accesses: otherwise the
     * pusher cannot see this reader and keeps reacquiring the unfair SRW
     * lock at word boundaries, delaying both the ISR and the next tick. */
    InterlockedIncrement(&s_waiters);
    AcquireSRWLockShared(&s_lock);
    InterlockedDecrement(&s_waiters);
    up = line_up();
    ReleaseSRWLockShared(&s_lock);
    return up;
}

void *kelvin_irq_event(void)
{
    return InterlockedCompareExchange(&s_trapped, 0, 0) ? s_irq_event : NULL;
}

void kelvin_vblank(void)
{
    if (!InterlockedCompareExchange(&s_trapped, 0, 0)) {
        return;
    }
    InterlockedIncrement(&s_waiters);
    kelvin_lock();
    InterlockedDecrement(&s_waiters);
    s_pcrtc[NV_PCRTC_INTR_0 / 4] |= NV_PCRTC_INTR_0_VBLANK;
    kelvin_irq_update();
    kelvin_unlock();
}

void kelvin_context_switch(KelvinState *pg, unsigned int chid)
{
    static unsigned said;
    bool channel_valid =
        pgraph_reg_r(pg, NV_PGRAPH_CTX_CONTROL) & NV_PGRAPH_CTX_CONTROL_CHID;
    unsigned int current = PG_GET_MASK(NV_PGRAPH_CTX_USER,
                                       NV_PGRAPH_CTX_USER_CHID);

    if (channel_valid && current == chid) {
        return;
    }
    if (said++ < 4) {
        fprintf(stderr, "[KELVIN] context switch to channel %u requested"
                        " (CTX_CONTROL 0x%08X, CTX_USER 0x%08X)\n", chid,
                pgraph_reg_r(pg, NV_PGRAPH_CTX_CONTROL),
                pgraph_reg_r(pg, NV_PGRAPH_CTX_USER));
        fflush(stderr);
    }
    PG_SET_MASK(NV_PGRAPH_TRAPPED_ADDR, NV_PGRAPH_TRAPPED_ADDR_CHID, chid);
    pg->waiting_for_context_switch = true;
    pg->pending_interrupts |= NV_PGRAPH_INTR_CONTEXT_SWITCH;
    kelvin_irq_update();
}

/* ---- PMC ------------------------------------------------------------------------ */

static uint32_t pmc_read(uint32_t reg)
{
    switch (reg) {
    case NV_PMC_BOOT_0:
        return 0x02A000A3;           /* NV2A, A03, rev 0 -- as xemu reports */
    case NV_PMC_INTR_0:
        return pmc_pending();
    default:
        return s_pmc[reg / 4];
    }
}

static void pmc_write(uint32_t reg, uint32_t val, uint32_t mask)
{
    switch (reg) {
    case NV_PMC_INTR_0:
        /* Write-1-to-clear, but every bit here is computed from a unit's
         * own status: clearing it means clearing that. */
        kelvin_irq_update();
        break;
    case NV_PMC_INTR_EN_0:
        s_pmc[reg / 4] = (s_pmc[reg / 4] & ~mask) | (val & mask);
        kelvin_irq_update();
        break;
    default:
        s_pmc[reg / 4] = (s_pmc[reg / 4] & ~mask) | (val & mask);
        break;
    }
}

/* ---- PCRTC ---------------------------------------------------------------------- */

static uint32_t pcrtc_read(uint32_t reg)
{
    if (reg == NV_PCRTC_RASTER) {
        return s_pcrtc_raster++;
    }
    return s_pcrtc[reg / 4];
}

static void pcrtc_write(uint32_t reg, uint32_t val, uint32_t mask)
{
    switch (reg) {
    case NV_PCRTC_INTR_0:
        s_pcrtc[reg / 4] &= ~(val & mask);
        kelvin_irq_update();
        break;
    case NV_PCRTC_INTR_EN_0:
        s_pcrtc[reg / 4] = (s_pcrtc[reg / 4] & ~mask) | (val & mask);
        kelvin_irq_update();
        break;
    case NV_PCRTC_START: {
        static unsigned said;
        uint32_t start = ((s_pcrtc[reg / 4] & ~mask) | (val & mask))
                         & 0x07FFFFFF;
        if (start != s_pcrtc[reg / 4] && said++ < 8) {
            fprintf(stderr, "[KELVIN] scanout (PCRTC_START) -> 0x%08X\n",
                    start);
            fflush(stderr);
        }
        s_pcrtc[reg / 4] = start;
        break;
    }
    default:
        s_pcrtc[reg / 4] = (s_pcrtc[reg / 4] & ~mask) | (val & mask);
        break;
    }
}

/* ---- PGRAPH --------------------------------------------------------------------- */

static uint32_t rdi_access(KelvinState *pg, bool write, uint32_t val)
{
    unsigned int select = PG_GET_MASK(NV_PGRAPH_RDI_INDEX,
                                      NV_PGRAPH_RDI_INDEX_SELECT);
    unsigned int address = PG_GET_MASK(NV_PGRAPH_RDI_INDEX,
                                       NV_PGRAPH_RDI_INDEX_ADDRESS);
    uint32_t r = 0;

    if ((select == RDI_INDEX_VTX_CONSTANTS0 ||
         select == RDI_INDEX_VTX_CONSTANTS1) &&
        address / 4 < NV2A_VERTEXSHADER_CONSTANTS) {
        uint32_t *c = &pg->vsh_constants[address / 4][3 - address % 4];
        if (write) {
            pg->vsh_constants_dirty[address / 4] |= (val != *c);
            *c = val;
        } else {
            r = *c;
        }
    } else {
        /* xemu does not model other RAM selects either: reads are zero and
         * writes vanish. */
        static unsigned said;
        if (said++ < 4) {
            fprintf(stderr, "[KELVIN] RDI %s select 0x%X address 0x%X"
                            " not modelled\n", write ? "write" : "read",
                    select, address);
            fflush(stderr);
        }
    }
    PG_SET_MASK(NV_PGRAPH_RDI_INDEX, NV_PGRAPH_RDI_INDEX_ADDRESS,
                address + 1);
    return r;
}

static uint32_t pgraph_read(uint32_t reg)
{
    KelvinState *pg = s_pg;

    switch (reg) {
    case NV_PGRAPH_INTR:
        return pg->pending_interrupts;
    case NV_PGRAPH_INTR_EN:
        return pg->enabled_interrupts;
    case NV_PGRAPH_RDI_DATA:
        return rdi_access(pg, false, 0);
    default:
        return pgraph_reg_r(pg, reg);
    }
}

static void pgraph_write(uint32_t reg, uint32_t val, uint32_t mask)
{
    KelvinState *pg = s_pg;

    switch (reg) {
    case NV_PGRAPH_INTR:
        pg->pending_interrupts &= ~(val & mask);
        if (!(pg->pending_interrupts & NV_PGRAPH_INTR_ERROR)) {
            if (pg->waiting_for_nop) {
                extern volatile long long g_kelvin_notify_raise_qpc, g_kelvin_notify_ticks, g_kelvin_notify_count;
                LARGE_INTEGER now;
                QueryPerformanceCounter(&now);
                g_kelvin_notify_ticks += now.QuadPart - g_kelvin_notify_raise_qpc;
                g_kelvin_notify_count++;
            }
            pg->waiting_for_nop = false;
        }
        if (!(pg->pending_interrupts & NV_PGRAPH_INTR_CONTEXT_SWITCH)) {
            pg->waiting_for_context_switch = false;
        }
        kelvin_irq_update();
        break;
    case NV_PGRAPH_INTR_EN:
        pg->enabled_interrupts = (pg->enabled_interrupts & ~mask)
                                 | (val & mask);
        kelvin_irq_update();
        break;
    case NV_PGRAPH_INCREMENT:
        /* A trigger: the vblank handler advancing the flip read index. */
        if (val & mask & NV_PGRAPH_INCREMENT_READ_3D) {
            uint32_t modulo = PG_GET_MASK(NV_PGRAPH_SURFACE,
                                          NV_PGRAPH_SURFACE_MODULO_3D);
            uint32_t next = PG_GET_MASK(NV_PGRAPH_SURFACE,
                                        NV_PGRAPH_SURFACE_READ_3D) + 1;
            if (modulo) {
                next %= modulo;
            }
            PG_SET_MASK(NV_PGRAPH_SURFACE, NV_PGRAPH_SURFACE_READ_3D, next);
        }
        break;
    case NV_PGRAPH_RDI_DATA:
        rdi_access(pg, true, val);
        break;
    case NV_PGRAPH_CHANNEL_CTX_TRIGGER:
        if (val & mask & NV_PGRAPH_CHANNEL_CTX_TRIGGER_READ_IN) {
            uint32_t ctx = PG_GET_MASK(NV_PGRAPH_CHANNEL_CTX_POINTER,
                                       NV_PGRAPH_CHANNEL_CTX_POINTER_INST)
                           << 4;
            pgraph_reg_w(pg, NV_PGRAPH_CTX_USER, kelvin_pramin_r32(ctx));
        }
        break;
    default:
        pgraph_reg_w(pg, reg, (pgraph_reg_r(pg, reg) & ~mask) | (val & mask));
        break;
    }
}

/* ---- dispatch ------------------------------------------------------------------- */

static bool owned(uint32_t off)
{
    return off - PMC_BASE < PMC_SIZE ||
           off - KELVIN_PGRAPH_BASE < PGRAPH_SIZE ||
           off - PCRTC_BASE < PCRTC_SIZE ||
           off - PRMDIO_BASE < PRMDIO_SIZE;
}

int kelvin_owns_register(unsigned int offset)
{
    /* Unowned blocks never need a state query. The trap publisher uses an
     * InterlockedExchange after initialization; keep every owned query fresh
     * without taking exclusive ownership of the flag's cache line. */
    if (!owned(offset)) {
        return 0;
    }
#ifdef _MSC_VER
    return ReadAcquire(&s_trapped) != 0;
#else
    return __atomic_load_n(&s_trapped, __ATOMIC_ACQUIRE) != 0;
#endif
}

static uint32_t reg_read(uint32_t reg)
{
    if (reg - PMC_BASE < PMC_SIZE) {
        return pmc_read(reg - PMC_BASE);
    }
    if (reg - KELVIN_PGRAPH_BASE < PGRAPH_SIZE) {
        return pgraph_read(reg - KELVIN_PGRAPH_BASE);
    }
    return pcrtc_read(reg - PCRTC_BASE);
}

/* `mask` names the bytes the access covers: a narrow write to a storage
 * register merges, and one to a write-1-to-clear register clears only what
 * it wrote. */
static void reg_write(uint32_t reg, uint32_t val, uint32_t mask)
{
    if (reg - PMC_BASE < PMC_SIZE) {
        pmc_write(reg - PMC_BASE, val, mask);
    } else if (reg - KELVIN_PGRAPH_BASE < PGRAPH_SIZE) {
        pgraph_write(reg - KELVIN_PGRAPH_BASE, val, mask);
    } else {
        pcrtc_write(reg - PCRTC_BASE, val, mask);
    }
}

static uint64_t access_read(void *dev, uint32_t off, int size)
{
    unsigned shift = (off & 3) * 8;
    uint64_t v;
    (void)dev;

    if (off - PRMDIO_BASE < PRMDIO_SIZE) {
        /* byte registers, and reading 0x3C9 advances the palette: only the bytes accessed */
        int i;
        for (v = 0, i = 0; i < size; i++) v |= (uint64_t)nv2a_dac_read(off - PRMDIO_BASE + i) << (i * 8);
        return v;
    }
    if (size == 8) {
        return reg_read(off & ~3u) | ((uint64_t)reg_read((off & ~3u) + 4) << 32);
    }
    v = reg_read(off & ~3u) >> shift;
    return size >= 4 ? v & 0xFFFFFFFFu : v & ((1u << (size * 8)) - 1);
}

static void access_write(void *dev, uint32_t off, uint64_t val, int size)
{
    unsigned shift = (off & 3) * 8;
    uint32_t mask;
    (void)dev;

    if (off - PRMDIO_BASE < PRMDIO_SIZE) {
        int i;
        for (i = 0; i < size; i++) nv2a_dac_write(off - PRMDIO_BASE + i, (uint8_t)(val >> (i * 8)));
        return;
    }
    if (size == 8) {
        reg_write(off & ~3u, (uint32_t)val, 0xFFFFFFFFu);
        reg_write((off & ~3u) + 4, (uint32_t)(val >> 32), 0xFFFFFFFFu);
        return;
    }
    mask = size >= 4 ? 0xFFFFFFFFu : ((1u << (size * 8)) - 1);
    reg_write(off & ~3u, (uint32_t)(val << shift), mask << shift);
}

extern void kgpu_symbol_name(uint64_t addr, char *out, size_t cap);

static LONG CALLBACK kelvin_mmio_veh(PEXCEPTION_POINTERS ep)
{
    const EXCEPTION_RECORD *er = ep->ExceptionRecord;
    uintptr_t fault;
    uint32_t off;
    int done;

    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        er->NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    fault = (uintptr_t)er->ExceptionInformation[1];
    if (fault < (uintptr_t)s_ap || fault - (uintptr_t)s_ap >= 0x01000000u) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    off = (uint32_t)(fault - (uintptr_t)s_ap);
    if (!owned(off)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    {
        /* RECOMP_KELVIN_MMIOSTAT=1: once a second, which registers were touched how often and by which thread */
        static int stat_on = -1;
        static struct { uint32_t off; DWORD tid; uint64_t rip; unsigned n; } tab[64];
        static ULONGLONG last_tick;
        if (stat_on < 0) { const char *e = getenv("RECOMP_KELVIN_MMIOSTAT"); stat_on = e && *e && *e != '0'; }
        if (stat_on) {
            unsigned k;
            ULONGLONG now = GetTickCount64();
            DWORD tid = GetCurrentThreadId();
            for (k = 0; k < 64 && tab[k].n; k++) if (tab[k].off == off && tab[k].tid == tid && tab[k].rip == ep->ContextRecord->Rip) break;
            if (k < 64) { tab[k].off = off; tab[k].tid = tid; tab[k].rip = ep->ContextRecord->Rip; tab[k].n++; }
            if (!last_tick) last_tick = now;
            if (now - last_tick >= 1000) {
                fprintf(stderr, "[KMMIO] last second:");
                for (k = 0; k < 64 && tab[k].n; k++) { char nm[64]; kgpu_symbol_name(tab[k].rip, nm, sizeof(nm)); fprintf(stderr, " 0x%X/tid%lu/%s x%u", tab[k].off, (unsigned long)tab[k].tid, nm, tab[k].n); }
                fprintf(stderr, "\n");
                memset(tab, 0, sizeof(tab));
                last_tick = now;
            }
        }
    }
    InterlockedIncrement(&s_waiters);
    kelvin_lock();
    InterlockedDecrement(&s_waiters);
    done = mmio_emulate(ep->ContextRecord, off, NULL, access_read,
                        access_write);
    kelvin_unlock();
    if (done) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    {
        const uint8_t *ip = (const uint8_t *)ep->ContextRecord->Rip;
        fprintf(stderr, "[KELVIN] register access at 0x%08X by an"
                        " instruction the decoder does not know: %02X %02X"
                        " %02X %02X %02X %02X (host RIP %p)\n",
                KELVIN_APERTURE_VA + off, ip[0], ip[1], ip[2], ip[3], ip[4],
                ip[5], (void *)ip);
        fflush(stderr);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

bool kelvin_mmio_start(KelvinState *pg)
{
    static const struct { uint32_t base, size; uint32_t *shadow; } blocks[] = {
        { PMC_BASE, PMC_SIZE, s_pmc },
        { KELVIN_PGRAPH_BASE, PGRAPH_SIZE, s_pgraph },
        { PCRTC_BASE, PCRTC_SIZE, s_pcrtc },
        { PRMDIO_BASE, PRMDIO_SIZE, s_prmdio },
    };
    size_t i;
    DWORD old;

    s_pg = pg;
    s_ap = kelvin_aperture(0);
    for (i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++) {
        memcpy(blocks[i].shadow, s_ap + blocks[i].base, blocks[i].size);
    }
    /* As plain memory nothing could have been pending; what an ack thread
     * may have left in the status words is not an interrupt. */
    s_pmc[NV_PMC_INTR_0 / 4] = 0;
    s_pcrtc[NV_PCRTC_INTR_0 / 4] = 0;
    pg->regs = s_pgraph;
    pg->pending_interrupts = 0;
    pg->enabled_interrupts = s_pgraph[NV_PGRAPH_INTR_EN / 4];

    s_irq_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    /* First, so it runs before the game target's own crash reporter. */
    s_veh = AddVectoredExceptionHandler(1, kelvin_mmio_veh);
    if (!s_irq_event || !s_veh) {
        goto fail;
    }
    for (i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++) {
        if (!VirtualProtect(s_ap + blocks[i].base, blocks[i].size,
                            PAGE_NOACCESS, &old)) {
            while (i--) {
                VirtualProtect(s_ap + blocks[i].base, blocks[i].size,
                               PAGE_READWRITE, &old);
            }
            goto fail;
        }
    }
    InterlockedExchange(&s_trapped, 1);
    fprintf(stderr, "[KELVIN] PMC, PGRAPH, PCRTC and palette registers trapped"
                    " (interrupt status, enables and triggers modelled)\n");
    fflush(stderr);
    return true;

fail:
    fprintf(stderr, "[KELVIN] could not trap the NV2A registers (error %lu)\n",
            GetLastError());
    fflush(stderr);
    if (s_veh) {
        RemoveVectoredExceptionHandler(s_veh);
        s_veh = NULL;
    }
    if (s_irq_event) {
        CloseHandle(s_irq_event);
        s_irq_event = NULL;
    }
    return false;
}

void kelvin_mmio_report(void)
{
    KelvinState *pg = s_pg;

    if (!pg) {
        return;
    }
    AcquireSRWLockShared(&s_lock);
    fprintf(stderr, "[KELVIN]   interrupts: raised %llu; PMC_INTR_EN_0 %u,"
                    " PGRAPH pending 0x%08X enabled 0x%08X, PCRTC pending %u"
                    " enabled %u, FIFO access %u, scanout 0x%08X\n",
            (unsigned long long)s_irqs_raised, s_pmc[NV_PMC_INTR_EN_0 / 4],
            pg->pending_interrupts, pg->enabled_interrupts,
            s_pcrtc[NV_PCRTC_INTR_0 / 4], s_pcrtc[NV_PCRTC_INTR_EN_0 / 4],
            pgraph_reg_r(pg, NV_PGRAPH_FIFO) & NV_PGRAPH_FIFO_ACCESS,
            s_pcrtc[NV_PCRTC_START / 4]);
    ReleaseSRWLockShared(&s_lock);
}
