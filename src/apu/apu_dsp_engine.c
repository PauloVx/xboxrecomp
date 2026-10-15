#include "apu_dsp_engine.h"

#include <string.h>

static void set_buffer_region(Dsp56300MemoryRegion *region, uint32_t *base,
                              uint32_t words)
{
    region->start = 0;
    region->end = words;
    region->kind = DSP56300_REGION_BUFFER;
    region->data.buffer.base = base;
    region->data.buffer.offset = 0;
}

static void set_callback_region(Dsp56300MemoryRegion *region, uint32_t start,
                                uint32_t end,
                                const Dsp56300CallbackRegion *callback)
{
    region->start = start;
    region->end = end;
    region->kind = DSP56300_REGION_CALLBACK;
    region->data.callback = *callback;
}

static bool copy_regions(Dsp56300MemoryRegion *destination,
                         const Dsp56300MemoryRegion *source, uint32_t count)
{
    uint32_t i, j;
    if (count > MCPX_DSP_MAX_MEMORY_REGIONS || (count && !source)) return false;
    for (i = 0; i < count; ++i) {
        if (source[i].start >= source[i].end) return false;
        if (source[i].kind == DSP56300_REGION_BUFFER &&
            !source[i].data.buffer.base) return false;
        if (source[i].kind == DSP56300_REGION_CALLBACK &&
            (!source[i].data.callback.read || !source[i].data.callback.write))
            return false;
        for (j = 0; j < i; ++j) {
            if (source[i].start < source[j].end &&
                source[j].start < source[i].end) return false;
        }
        destination[i] = source[i];
    }
    return true;
}

bool mcpx_dsp_engine_init_with_map(McpxDspEngine *engine,
                                   const Dsp56300MemoryMap *memory_map)
{
    if (!engine) return false;
    memset(engine, 0, sizeof(*engine));
    if (!memory_map ||
        !copy_regions(engine->x_regions, memory_map->x_regions,
                      memory_map->x_count) ||
        !copy_regions(engine->y_regions, memory_map->y_regions,
                      memory_map->y_count) ||
        !copy_regions(engine->p_regions, memory_map->p_regions,
                      memory_map->p_count))
        return false;

    engine->memory_map.x_regions = engine->x_regions;
    engine->memory_map.x_count = memory_map->x_count;
    engine->memory_map.y_regions = engine->y_regions;
    engine->memory_map.y_count = memory_map->y_count;
    engine->memory_map.p_regions = engine->p_regions;
    engine->memory_map.p_count = memory_map->p_count;
    engine->create_info.memory_map = engine->memory_map;
    engine->jit = dsp56300_create(&engine->create_info);
    return engine->jit != NULL;
}

bool mcpx_dsp_engine_init(McpxDspEngine *engine,
                          uint32_t *xram, uint32_t x_words,
                          uint32_t *yram, uint32_t y_words,
                          uint32_t *pram, uint32_t p_words)
{
    return mcpx_dsp_engine_init_with_peripherals(
        engine, xram, x_words, yram, y_words, pram, p_words,
        0, 0, NULL, NULL);
}

bool mcpx_dsp_engine_init_with_peripherals(
    McpxDspEngine *engine,
    uint32_t *xram, uint32_t x_words,
    uint32_t *yram, uint32_t y_words,
    uint32_t *pram, uint32_t p_words,
    uint32_t peripheral_start, uint32_t peripheral_end,
    const Dsp56300CallbackRegion *x_peripheral,
    const Dsp56300CallbackRegion *y_peripheral)
{
    Dsp56300MemoryRegion x_regions[2], y_regions[2], p_regions[1];
    Dsp56300MemoryMap memory_map = {0};
    uint32_t x_count = 1, y_count = 1;
    if (!engine || !xram || !yram || !pram || !x_words || !y_words || !p_words)
        return false;
    if ((x_peripheral || y_peripheral) &&
        (peripheral_start >= peripheral_end ||
         (x_peripheral && (!x_peripheral->read || !x_peripheral->write)) ||
         (y_peripheral && (!y_peripheral->read || !y_peripheral->write))))
        return false;

    set_buffer_region(&x_regions[0], xram, x_words);
    set_buffer_region(&y_regions[0], yram, y_words);
    if (x_peripheral) {
        set_callback_region(&x_regions[x_count++], peripheral_start,
                            peripheral_end, x_peripheral);
    }
    if (y_peripheral) {
        set_callback_region(&y_regions[y_count++], peripheral_start,
                            peripheral_end, y_peripheral);
    }
    set_buffer_region(&p_regions[0], pram, p_words);
    memory_map.x_regions = x_regions;
    memory_map.x_count = x_count;
    memory_map.y_regions = y_regions;
    memory_map.y_count = y_count;
    memory_map.p_regions = p_regions;
    memory_map.p_count = 1;
    return mcpx_dsp_engine_init_with_map(engine, &memory_map);
}

void mcpx_dsp_engine_destroy(McpxDspEngine *engine)
{
    if (!engine) return;
    dsp56300_destroy(engine->jit);
    memset(engine, 0, sizeof(*engine));
}

void mcpx_dsp_engine_reset(McpxDspEngine *engine)
{
    if (engine && engine->jit) dsp56300_reset(engine->jit);
}

void mcpx_dsp_engine_step(McpxDspEngine *engine)
{
    if (engine && engine->jit) dsp56300_step(engine->jit);
}

void mcpx_dsp_engine_run(McpxDspEngine *engine, int32_t cycles)
{
    if (engine && engine->jit && cycles > 0)
        dsp56300_run(engine->jit, cycles);
}

void mcpx_dsp_engine_invalidate_cache(McpxDspEngine *engine)
{
    if (engine && engine->jit) dsp56300_invalidate_cache(engine->jit);
}

bool mcpx_dsp_engine_get_pc(const McpxDspEngine *engine, uint32_t *pc)
{
    Dsp56300State state;
    if (!engine || !engine->jit || !pc) return false;
    dsp56300_get_state(engine->jit, &state);
    *pc = state.pc;
    return true;
}

static bool to_ffi_space(McpxDspMemorySpace space, Dsp56300MemSpace *out)
{
    if (!out) return false;
    switch (space) {
    case MCPX_DSP_SPACE_X: *out = DSP56300_MEM_SPACE_X; return true;
    case MCPX_DSP_SPACE_Y: *out = DSP56300_MEM_SPACE_Y; return true;
    case MCPX_DSP_SPACE_P: *out = DSP56300_MEM_SPACE_P; return true;
    default: return false;
    }
}

bool mcpx_dsp_engine_read(const McpxDspEngine *engine,
                          McpxDspMemorySpace space, uint32_t address,
                          uint32_t *value)
{
    Dsp56300MemSpace ffi_space;
    if (!engine || !engine->jit || !value || !to_ffi_space(space, &ffi_space))
        return false;
    *value = dsp56300_read_memory(engine->jit, ffi_space, address);
    return true;
}

bool mcpx_dsp_engine_write(McpxDspEngine *engine,
                           McpxDspMemorySpace space, uint32_t address,
                           uint32_t value)
{
    Dsp56300MemSpace ffi_space;
    if (!engine || !engine->jit || !to_ffi_space(space, &ffi_space))
        return false;
    dsp56300_write_memory(engine->jit, ffi_space, address, value & 0x00ffffffu);
    return true;
}
