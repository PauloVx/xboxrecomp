/*
 * Four vertices at once through one vertex program (nv2a_vsh_emulator4.h).
 *
 * Every operation follows nv2a_vsh_emulator.c and nv2a_vsh_cpu.c exactly:
 * operands are fetched, swizzled and multiplied by +-1 as fetch_value does,
 * both operations of a step read their inputs before either writes, and the
 * arithmetic is the same IEEE single operations in the same order, so each
 * lane's result is bit for bit the scalar emulator's. The common MAC
 * operations run on all four lanes in SSE; the rest call the scalar
 * functions lane by lane.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#include "nv2a_vsh_emulator4.h"

#include <emmintrin.h>
#include <string.h>

#include "nv2a_vsh_cpu.h"

static const Nv2aVshCpuFunc kScalar[] = {
    NULL,
    nv2a_vsh_cpu_mov,
    nv2a_vsh_cpu_mul,
    nv2a_vsh_cpu_add,
    nv2a_vsh_cpu_mad,
    nv2a_vsh_cpu_dp3,
    nv2a_vsh_cpu_dph,
    nv2a_vsh_cpu_dp4,
    nv2a_vsh_cpu_dst,
    nv2a_vsh_cpu_min,
    nv2a_vsh_cpu_max,
    nv2a_vsh_cpu_slt,
    nv2a_vsh_cpu_sge,
    nv2a_vsh_cpu_arl,
    nv2a_vsh_cpu_rcp,
    nv2a_vsh_cpu_rcc,
    nv2a_vsh_cpu_rsq,
    nv2a_vsh_cpu_exp,
    nv2a_vsh_cpu_log,
    nv2a_vsh_cpu_lit,
};

void nv2a_vsh_emu4_reset(Nv2aVshState4 *state) {
  __m128 zero = _mm_setzero_ps(), one = _mm_set1_ps(1.0f);
  int i;
  for (i = 0; i < 12; ++i) {
    state->temp[i].c[0] = state->temp[i].c[1] = zero;
    state->temp[i].c[2] = state->temp[i].c[3] = zero;
  }
  for (i = 0; i < 13; ++i) {
    state->output[i].c[0] = state->output[i].c[1] = zero;
    state->output[i].c[2] = zero;
    state->output[i].c[3] = one;
  }
  state->address[0] = state->address[1] = 0.0f;
  state->address[2] = state->address[3] = 0.0f;
}

/* fetch_value: register, relative context address, swizzle, and the
 * multiply by +1 or -1 that set_register always does. */
static void fetch4(const Nv2aVshState4 *state, const Nv2aVshInput *input,
                   Nv2aVshReg4 *out) {
  Nv2aVshReg4 gathered;
  const Nv2aVshReg4 *in;
  __m128 mult = _mm_set1_ps(input->is_negated ? -1.0f : 1.0f);
  int k;

  switch (input->type) {
    case NV2ART_INPUT:
      in = &state->input[input->index];
      break;
    case NV2ART_CONTEXT: {
      float v[4][4];
      int lane;
      if (!input->is_relative) {
        /* The same register for every lane: each component broadcast. */
        __m128 c = _mm_loadu_ps(&state->context[input->index * 4]);
        gathered.c[0] = _mm_shuffle_ps(c, c, 0x00);
        gathered.c[1] = _mm_shuffle_ps(c, c, 0x55);
        gathered.c[2] = _mm_shuffle_ps(c, c, 0xAA);
        gathered.c[3] = _mm_shuffle_ps(c, c, 0xFF);
        in = &gathered;
        break;
      }
      for (lane = 0; lane < 4; ++lane) {
        uint32_t offset = input->index;
        if (input->is_relative) {
          offset += (int)state->address[lane];
        }
        memcpy(v[lane], &state->context[offset * 4], sizeof(v[lane]));
      }
      for (k = 0; k < 4; ++k) {
        gathered.c[k] = _mm_setr_ps(v[0][k], v[1][k], v[2][k], v[3][k]);
      }
      in = &gathered;
      break;
    }
    default: /* NV2ART_TEMPORARY, and what the scalar code falls into */
      in = input->index == 12 ? &state->output[0] : &state->temp[input->index];
      break;
  }
  for (k = 0; k < 4; ++k) {
    out->c[k] = _mm_mul_ps(mult, in->c[input->swizzle[k]]);
  }
}

static __m128 bits(uint32_t v) { return _mm_castsi128_ps(_mm_set1_epi32((int)v)); }

static __m128 select4(__m128 mask, __m128 a, __m128 b) {
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
}

/* |x| is infinite. */
static __m128 is_inf4(__m128 x) {
  return _mm_cmpeq_ps(_mm_and_ps(x, bits(0x7FFFFFFF)), bits(0x7F800000));
}

/* The largest finite value of an infinity's sign, as fix_inf builds it. */
static __m128 finite_of4(__m128 x) {
  return _mm_castsi128_ps(_mm_add_epi32(
      _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32((int)0xFF000000)),
      _mm_set1_epi32(0x7FFFFF)));
}

/* fix_inf_mult */
static __m128 mul4(__m128 a, __m128 b) {
  __m128 zero = _mm_setzero_ps();
  __m128 either_zero = _mm_or_ps(_mm_cmpeq_ps(a, zero), _mm_cmpeq_ps(b, zero));
  __m128 p = _mm_mul_ps(a, b);
  __m128 fix = _mm_andnot_ps(_mm_or_ps(is_inf4(a), is_inf4(b)), is_inf4(p));
  p = select4(fix, finite_of4(p), p);
  return _mm_andnot_ps(either_zero, p);
}

/* fix_inf */
static __m128 fix_inf4(__m128 x) { return select4(is_inf4(x), finite_of4(x), x); }

/* nv2a_less_than: 1 or 0, with -0 below +0. */
static __m128 less4(__m128 a, __m128 b) {
  __m128i ai = _mm_castps_si128(a), bi = _mm_castps_si128(b);
  __m128 neg_zero = _mm_castsi128_ps(_mm_and_si128(
      _mm_cmpeq_epi32(ai, _mm_set1_epi32((int)0x80000000)),
      _mm_cmpeq_epi32(bi, _mm_setzero_si128())));
  return _mm_and_ps(_mm_or_ps(_mm_cmplt_ps(a, b), neg_zero), _mm_set1_ps(1.0f));
}

/* One operation on all lanes: SSE for the MAC arithmetic, the scalar
 * function lane by lane for the rest. */
static void operate4(Nv2aVshOpcode opcode, const Nv2aVshReg4 in[3],
                     Nv2aVshReg4 *r) {
  int k;
  switch (opcode) {
    case NV2AOP_MOV:
      *r = in[0];
      return;
    case NV2AOP_MUL:
      for (k = 0; k < 4; ++k) r->c[k] = mul4(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_ADD:
      for (k = 0; k < 4; ++k) r->c[k] = _mm_add_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_MAD:
      for (k = 0; k < 4; ++k) {
        r->c[k] = _mm_add_ps(mul4(in[0].c[k], in[1].c[k]), in[2].c[k]);
      }
      return;
    case NV2AOP_DP3:
    case NV2AOP_DPH:
    case NV2AOP_DP4: {
      __m128 s = _mm_add_ps(mul4(in[0].c[0], in[1].c[0]),
                            mul4(in[0].c[1], in[1].c[1]));
      s = _mm_add_ps(s, mul4(in[0].c[2], in[1].c[2]));
      if (opcode == NV2AOP_DPH) {
        s = _mm_add_ps(s, in[1].c[3]);
      } else if (opcode == NV2AOP_DP4) {
        s = _mm_add_ps(s, mul4(in[0].c[3], in[1].c[3]));
      }
      s = fix_inf4(s);
      r->c[0] = r->c[1] = r->c[2] = r->c[3] = s;
      return;
    }
    case NV2AOP_DST:
      r->c[0] = _mm_set1_ps(1.0f);
      r->c[1] = mul4(in[0].c[1], in[1].c[1]);
      r->c[2] = in[0].c[2];
      r->c[3] = in[1].c[3];
      return;
    case NV2AOP_MIN:
      /* a < b ? a : b */
      for (k = 0; k < 4; ++k) r->c[k] = _mm_min_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_MAX:
      /* a > b ? a : b */
      for (k = 0; k < 4; ++k) r->c[k] = _mm_max_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_SLT:
      for (k = 0; k < 4; ++k) r->c[k] = less4(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_SGE:
      for (k = 0; k < 4; ++k) {
        r->c[k] = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.0f),
                                        less4(in[0].c[k], in[1].c[k])),
                             _mm_set1_ps(1.0f));
      }
      return;
    default: {
      float lanes_in[3][4][4], lanes_out[4][4];
      int j, lane;
      for (j = 0; j < 3; ++j) {
        for (k = 0; k < 4; ++k) {
          float v[4];
          _mm_storeu_ps(v, in[j].c[k]);
          for (lane = 0; lane < 4; ++lane) lanes_in[j][lane][k] = v[lane];
        }
      }
      for (lane = 0; lane < 4; ++lane) {
        float inputs[12];
        for (j = 0; j < 3; ++j) memcpy(&inputs[j * 4], lanes_in[j][lane], 16);
        kScalar[opcode](lanes_out[lane], inputs);
      }
      for (k = 0; k < 4; ++k) {
        r->c[k] = _mm_setr_ps(lanes_out[0][k], lanes_out[1][k],
                              lanes_out[2][k], lanes_out[3][k]);
      }
      return;
    }
  }
}

/* apply_operation's writes. */
static void write4(Nv2aVshState4 *state, const Nv2aVshOperation *op,
                   const Nv2aVshReg4 *result) {
  const Nv2aVshOutput *out = op->outputs;
  int i;
  for (i = 0; i < 2; ++i, ++out) {
    Nv2aVshReg4 *reg;
    switch (out->type) {
      case NV2ART_OUTPUT:
        reg = &state->output[out->index];
        break;
      case NV2ART_TEMPORARY:
        if (out->index >= 12) {
          continue; /* r12 is read-only (o0); the scalar array has 12 */
        }
        reg = &state->temp[out->index];
        break;
      case NV2ART_ADDRESS: {
        /* ARL writes the scalar address register without a mask. */
        _mm_storeu_ps(state->address, result->c[0]);
        continue;
      }
      default: /* none, an input, or a context register (not allowed) */
        continue;
    }
    if (out->writemask & NV2AWM_X) reg->c[0] = result->c[0];
    if (out->writemask & NV2AWM_Y) reg->c[1] = result->c[1];
    if (out->writemask & NV2AWM_Z) reg->c[2] = result->c[2];
    if (out->writemask & NV2AWM_W) reg->c[3] = result->c[3];
  }
}

void nv2a_vsh_emu4_execute(Nv2aVshState4 *state, const Nv2aVshProgram *program) {
  const Nv2aVshStep *step = program->steps;
  for (;;) {
    Nv2aVshReg4 mac_in[3], ilu_in[3], result;
    int i;

    memset(mac_in, 0, sizeof(mac_in));
    memset(ilu_in, 0, sizeof(ilu_in));
    /* Both operations' inputs first, as prepare_inputs does. */
    if (step->mac.opcode) {
      for (i = 0; i < 3; ++i) {
        if (step->mac.inputs[i].type == NV2ART_NONE) {
          break;
        }
        fetch4(state, &step->mac.inputs[i], &mac_in[i]);
      }
    }
    if (step->ilu.opcode) {
      fetch4(state, &step->ilu.inputs[0], &ilu_in[0]);
    }
    if (step->mac.opcode) {
      operate4(step->mac.opcode, mac_in, &result);
      write4(state, &step->mac, &result);
    }
    if (step->ilu.opcode) {
      operate4(step->ilu.opcode, ilu_in, &result);
      write4(state, &step->ilu, &result);
    }
    if (step->is_final) {
      break;
    }
    ++step;
  }
}
