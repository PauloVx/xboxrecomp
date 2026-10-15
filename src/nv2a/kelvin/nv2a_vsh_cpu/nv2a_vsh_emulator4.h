/*
 * Four vertices at once through one vertex program: the scalar emulator's
 * semantics, operation for operation, with each register held as four SSE
 * vectors (x, y, z, w), one lane per vertex.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#ifndef NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR4_H_
#define NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR4_H_

#include <xmmintrin.h>

#include "nv2a_vsh_disassembler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A register of four vertices: c[k] holds component k of each lane. */
typedef struct Nv2aVshReg4 {
  __m128 c[4];
} Nv2aVshReg4;

typedef struct Nv2aVshState4 {
  Nv2aVshReg4 input[16];   /* v0-v15 */
  Nv2aVshReg4 output[13];  /* o0-o12 */
  Nv2aVshReg4 temp[12];    /* r0-r11; r12 reads o0 */
  float address[4];        /* a0 of each lane */
  /* c0-c191, shared by every lane, as the scalar emulator reads them. */
  const float *context;
} Nv2aVshState4;

/* Clear temporaries, set every output to (0, 0, 0, 1) and a0 to 0, as the
 * scalar driver does before each vertex. */
void nv2a_vsh_emu4_reset(Nv2aVshState4 *state);

/* Run the program on all four lanes. The program must not write context
 * registers: every lane would write the same shared array. */
void nv2a_vsh_emu4_execute(Nv2aVshState4 *state, const Nv2aVshProgram *program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR4_H_
