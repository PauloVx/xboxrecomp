/* C11 adapter for the MIT-licensed DSP56300 JIT library. */
#pragma once

#include "dsp56300.h"

#include <stdbool.h>
#include <stdint.h>

typedef enum McpxDspMemorySpace {
    MCPX_DSP_SPACE_X = DSP56300_MEM_SPACE_X,
    MCPX_DSP_SPACE_Y = DSP56300_MEM_SPACE_Y,
    MCPX_DSP_SPACE_P = DSP56300_MEM_SPACE_P
} McpxDspMemorySpace;

#define MCPX_DSP_MAX_MEMORY_REGIONS 4u

typedef struct McpxDspEngine {
    Dsp56300Jit *jit;
    Dsp56300MemoryRegion x_regions[MCPX_DSP_MAX_MEMORY_REGIONS];
    Dsp56300MemoryRegion y_regions[MCPX_DSP_MAX_MEMORY_REGIONS];
    Dsp56300MemoryRegion p_regions[MCPX_DSP_MAX_MEMORY_REGIONS];
    Dsp56300MemoryMap memory_map;
    Dsp56300CreateInfo create_info;
} McpxDspEngine;

bool mcpx_dsp_engine_init_with_map(McpxDspEngine *engine,
                                   const Dsp56300MemoryMap *memory_map);
bool mcpx_dsp_engine_init(McpxDspEngine *engine,
                          uint32_t *xram, uint32_t x_words,
                          uint32_t *yram, uint32_t y_words,
                          uint32_t *pram, uint32_t p_words);
bool mcpx_dsp_engine_init_with_peripherals(
    McpxDspEngine *engine,
    uint32_t *xram, uint32_t x_words,
    uint32_t *yram, uint32_t y_words,
    uint32_t *pram, uint32_t p_words,
    uint32_t peripheral_start, uint32_t peripheral_end,
    const Dsp56300CallbackRegion *x_peripheral,
    const Dsp56300CallbackRegion *y_peripheral);
void mcpx_dsp_engine_destroy(McpxDspEngine *engine);
void mcpx_dsp_engine_reset(McpxDspEngine *engine);
void mcpx_dsp_engine_step(McpxDspEngine *engine);
void mcpx_dsp_engine_run(McpxDspEngine *engine, int32_t cycles);
void mcpx_dsp_engine_invalidate_cache(McpxDspEngine *engine);
bool mcpx_dsp_engine_get_pc(const McpxDspEngine *engine, uint32_t *pc);
bool mcpx_dsp_engine_read(const McpxDspEngine *engine,
                          McpxDspMemorySpace space, uint32_t address,
                          uint32_t *value);
bool mcpx_dsp_engine_write(McpxDspEngine *engine,
                           McpxDspMemorySpace space, uint32_t address,
                           uint32_t value);
