#include "apu_dma.h"
#include "apu_dma_regs.h"

#include <string.h>

enum { TEST_WORDS = 0x1800, TEST_BYTES = 256 };

typedef struct TestDmaMemory {
    uint32_t words[TEST_WORDS];
    uint8_t bytes[TEST_BYTES];
} TestDmaMemory;

static uint32_t read_word(void *opaque, int space, uint32_t address)
{
    TestDmaMemory *memory = opaque;
    (void)space;
    return address < TEST_WORDS ? memory->words[address] : 0;
}

static void write_word(void *opaque, int space, uint32_t address,
                       uint32_t value)
{
    TestDmaMemory *memory = opaque;
    (void)space;
    if (address < TEST_WORDS) memory->words[address] = value;
}

static void read_words(void *opaque, int space, uint32_t address,
                       uint32_t *out, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
        out[i] = read_word(opaque, space, address + i);
}

static void write_words(void *opaque, int space, uint32_t address,
                        const uint32_t *values, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
        write_word(opaque, space, address + i, values[i]);
}

static bool scratch(void *opaque, uint8_t *bytes, uint32_t addr, size_t len,
                    bool dir)
{
    TestDmaMemory *memory = opaque;
    if (addr > TEST_BYTES || len > TEST_BYTES - addr) return false;
    if (dir) memcpy(memory->bytes + addr, bytes, len);
    else memcpy(bytes, memory->bytes + addr, len);
    return true;
}

static bool fifo(void *opaque, uint8_t *bytes, unsigned int index,
                 size_t len, bool dir)
{
    TestDmaMemory *memory = opaque;
    if (index != 0 || len > TEST_BYTES) return false;
    if (dir) memcpy(memory->bytes, bytes, len);
    else memcpy(bytes, memory->bytes, len);
    return true;
}

int test_dsp_dma_transfer(void)
{
    TestDmaMemory memory = {0};
    DSPDMAState dma = {0};
    dma.mem_opaque = &memory;
    dma.mem_read = read_word;
    dma.mem_write = write_word;
    dma.mem_read_run = read_words;
    dma.mem_write_run = write_words;
    dma.rw_opaque = &memory;
    dma.scratch_rw = scratch;
    dma.fifo_rw = fifo;
    /* One linear DSP-to-FIFO node, then EOL. Format 2 packs one 24-bit
     * DSP word into a 32-bit slot. */
    memory.words[0] = NODE_POINTER_EOL;
    memory.words[1] = (1u << 1) | (1u << 14) | (2u << 10);
    memory.words[2] = 2;
    memory.words[3] = 0x10;
    memory.words[4] = 0;
    memory.words[5] = 0;
    memory.words[6] = 0xFF;
    memory.words[0x10] = 0x123456;
    memory.words[0x11] = 0xABCDEF;

    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!dma.eol || dma.error || memory.bytes[1] != 0x56 ||
        memory.bytes[2] != 0x34 || memory.bytes[3] != 0x12 ||
        memory.bytes[5] != 0xEF || memory.bytes[6] != 0xCD ||
        memory.bytes[7] != 0xAB) {
        dsp_dma_destroy(&dma);
        return 1;
    }
    dsp_dma_completion_seen(&dma);
    if (dma.control & DMA_CONTROL_RUNNING) {
        dsp_dma_destroy(&dma);
        return 2;
    }

    /* Completed transfers retain two RUNNING polls for the startup
     * handshake, then report completion. Freezing holds that completion
     * until UNFREEZE regardless of how often the guest polls. */
    dsp_dma_reset(&dma);
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_FREEZE);
    for (unsigned i = 0; i < 5; ++i) {
        uint32_t control = dsp_dma_read(&dma, DMA_CONTROL);
        if ((control & (DMA_CONTROL_RUNNING | DMA_CONTROL_FROZEN)) !=
            (DMA_CONTROL_RUNNING | DMA_CONTROL_FROZEN) || dma.eol ||
            !dma.eol_held) {
            dsp_dma_destroy(&dma);
            return 6;
        }
    }
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_UNFREEZE);
    if (!dma.eol || dma.eol_held ||
        !(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING) ||
        !(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING) ||
        (dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
        dsp_dma_destroy(&dma);
        return 7;
    }

    /* EOL is a sticky interrupt latch. Starting a later cyclic chain must
     * not let that earlier latch retire the new transfer through polling
     * or through acknowledgement of the previous completion interrupt. */
    memory.words[0] = 0;
    memory.words[1] = (1u << 1) | (4u << 10);
    memory.words[2] = 0;
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!dma.eol) {
        dsp_dma_destroy(&dma);
        return 10;
    }
    for (unsigned i = 0; i < 5; ++i) {
        if (!(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
            dsp_dma_destroy(&dma);
            return 11;
        }
    }
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_FREEZE);
    dsp_dma_completion_seen(&dma);
    if (!(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
        dsp_dma_destroy(&dma);
        return 12;
    }
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_UNFREEZE);
    dsp_dma_completion_seen(&dma);
    if (!(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
        dsp_dma_destroy(&dma);
        return 13;
    }

    /* An empty chain is already at its terminator and also completes. */
    dsp_dma_reset(&dma);
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, NODE_POINTER_EOL);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING) ||
        !(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING) ||
        (dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
        dsp_dma_destroy(&dma);
        return 14;
    }

    /* An unsupported format is rejected, reported, and does not transfer. */
    dsp_dma_reset(&dma);
    memset(memory.bytes, 0, sizeof(memory.bytes));
    memory.words[0] = NODE_POINTER_EOL;
    memory.words[1] = (1u << 1) | (4u << 10);
    memory.words[2] = 4;
    memory.words[3] = 0x20;
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!dma.error || !(dma.pending_interrupts & DMA_INTERRUPT_ERROR) ||
        memory.bytes[0] != 0) {
        dsp_dma_destroy(&dma);
        return 3;
    }

    /* Failed guest reads must not unpack stale staging bytes into DSP RAM. */
    dsp_dma_reset(&dma);
    memory.words[0] = NODE_POINTER_EOL;
    memory.words[1] = (0u << 1) | (1u << 14) | (2u << 10);
    memory.words[2] = 100;
    memory.words[3] = 0x20;
    memory.words[0x20] = 0x654321;
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!dma.error || !(dma.pending_interrupts & DMA_INTERRUPT_ERROR) ||
        memory.words[0x20] != 0x654321) {
        dsp_dma_destroy(&dma);
        return 4;
    }

    /* A cyclic descriptor chain must remain bounded for this synchronous
     * backend, with RUNNING left visible as in the original model. */
    dsp_dma_reset(&dma);
    memory.words[0] = 0;
    memory.words[1] = (1u << 1) | (4u << 10);
    memory.words[2] = 0;
    memory.words[3] = 0;
    dsp_dma_write(&dma, DMA_NEXT_BLOCK, 0);
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_START);
    if (!(dma.control & DMA_CONTROL_RUNNING) || !dma.error) {
        dsp_dma_destroy(&dma);
        return 5;
    }
    for (unsigned i = 0; i < 5; ++i) {
        if (!(dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING)) {
            dsp_dma_destroy(&dma);
            return 8;
        }
    }
    dsp_dma_write(&dma, DMA_CONTROL, DMA_CONTROL_ACTION_STOP);
    if (dsp_dma_read(&dma, DMA_CONTROL) & DMA_CONTROL_RUNNING) {
        dsp_dma_destroy(&dma);
        return 9;
    }
    dsp_dma_destroy(&dma);
    return 0;
}
