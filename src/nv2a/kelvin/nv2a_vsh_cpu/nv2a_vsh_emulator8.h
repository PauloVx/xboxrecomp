/*
 * Eight vertices at once through one vertex program, on AVX2: the 4-lane
 * emulator (nv2a_vsh_emulator4.h) with 256-bit vectors, operation for
 * operation, so each lane's result is bit for bit the scalar emulator's.
 * Built only where the compiler targets AVX2 (NV2A_VSH_HAVE_EMU8).
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#ifndef NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR8_H_
#define NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR8_H_

#if defined(__AVX2__)
#define NV2A_VSH_HAVE_EMU8 1

#include <immintrin.h>

#include "nv2a_vsh_disassembler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A register of eight vertices: c[k] holds component k of each lane. */
typedef struct Nv2aVshReg8 {
  __m256 c[4];
} Nv2aVshReg8;

typedef struct Nv2aVshState8 {
  Nv2aVshReg8 input[16];   /* v0-v15 */
  Nv2aVshReg8 output[13];  /* o0-o12 */
  Nv2aVshReg8 temp[12];    /* r0-r11; r12 reads o0 */
  float address[8];        /* a0 of each lane */
  /* c0-c191, shared by every lane, as the scalar emulator reads them. */
  const float *context;
} Nv2aVshState8;

void nv2a_vsh_emu8_reset(Nv2aVshState8 *state);

/* Run the program on all eight lanes. The program must not write context
 * registers. */
void nv2a_vsh_emu8_execute(Nv2aVshState8 *state, const Nv2aVshProgram *program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __AVX2__
#endif  // NV2A_VSH_CPU_SRC_NV2A_VSH_EMULATOR8_H_
