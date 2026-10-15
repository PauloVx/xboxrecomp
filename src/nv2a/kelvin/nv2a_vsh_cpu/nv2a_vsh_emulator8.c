/*
 * Eight vertices at once through one vertex program (nv2a_vsh_emulator8.h).
 *
 * nv2a_vsh_emulator4.c on 256-bit vectors: the same operations in the same
 * order per lane, the same compare predicates (_mm_cmpeq_ps is EQ_OQ,
 * _mm_cmplt_ps is LT_OS), the scalar functions lane by lane for the rest.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#include "nv2a_vsh_emulator8.h"

#ifdef NV2A_VSH_HAVE_EMU8

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

void nv2a_vsh_emu8_reset(Nv2aVshState8 *state) {
  __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
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
  memset(state->address, 0, sizeof(state->address));
}

/* fetch_value: register, relative context address, swizzle, and the
 * multiply by +1 or -1 that set_register always does. */
static void fetch8(const Nv2aVshState8 *state, const Nv2aVshInput *input,
                   Nv2aVshReg8 *out) {
  Nv2aVshReg8 gathered;
  const Nv2aVshReg8 *in;
  __m256 mult = _mm256_set1_ps(input->is_negated ? -1.0f : 1.0f);
  int k;

  switch (input->type) {
    case NV2ART_INPUT:
      in = &state->input[input->index];
      break;
    case NV2ART_CONTEXT: {
      int lane;
      if (!input->is_relative) {
        /* The same register for every lane: each component broadcast. */
        const float *c = &state->context[input->index * 4];
        for (k = 0; k < 4; ++k) gathered.c[k] = _mm256_set1_ps(c[k]);
      } else {
        float v[4][8];
        for (lane = 0; lane < 8; ++lane) {
          uint32_t offset = input->index + (int)state->address[lane];
          const float *c = &state->context[offset * 4];
          for (k = 0; k < 4; ++k) v[k][lane] = c[k];
        }
        for (k = 0; k < 4; ++k) gathered.c[k] = _mm256_loadu_ps(v[k]);
      }
      in = &gathered;
      break;
    }
    default: /* NV2ART_TEMPORARY, and what the scalar code falls into */
      in = input->index == 12 ? &state->output[0] : &state->temp[input->index];
      break;
  }
  for (k = 0; k < 4; ++k) {
    out->c[k] = _mm256_mul_ps(mult, in->c[input->swizzle[k]]);
  }
}

static __m256 bits(uint32_t v) {
  return _mm256_castsi256_ps(_mm256_set1_epi32((int)v));
}

static __m256 select8(__m256 mask, __m256 a, __m256 b) {
  return _mm256_or_ps(_mm256_and_ps(mask, a), _mm256_andnot_ps(mask, b));
}

/* |x| is infinite. */
static __m256 is_inf8(__m256 x) {
  return _mm256_cmp_ps(_mm256_and_ps(x, bits(0x7FFFFFFF)), bits(0x7F800000),
                       _CMP_EQ_OQ);
}

/* The largest finite value of an infinity's sign, as fix_inf builds it. */
static __m256 finite_of8(__m256 x) {
  return _mm256_castsi256_ps(_mm256_add_epi32(
      _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32((int)0xFF000000)),
      _mm256_set1_epi32(0x7FFFFF)));
}

/* fix_inf_mult */
static __m256 mul8(__m256 a, __m256 b) {
  __m256 zero = _mm256_setzero_ps();
  __m256 either_zero = _mm256_or_ps(_mm256_cmp_ps(a, zero, _CMP_EQ_OQ),
                                    _mm256_cmp_ps(b, zero, _CMP_EQ_OQ));
  __m256 p = _mm256_mul_ps(a, b);
  __m256 fix = _mm256_andnot_ps(_mm256_or_ps(is_inf8(a), is_inf8(b)), is_inf8(p));
  p = select8(fix, finite_of8(p), p);
  return _mm256_andnot_ps(either_zero, p);
}

/* fix_inf */
static __m256 fix_inf8(__m256 x) { return select8(is_inf8(x), finite_of8(x), x); }

/* nv2a_less_than: 1 or 0, with -0 below +0. */
static __m256 less8(__m256 a, __m256 b) {
  __m256i ai = _mm256_castps_si256(a), bi = _mm256_castps_si256(b);
  __m256 neg_zero = _mm256_castsi256_ps(_mm256_and_si256(
      _mm256_cmpeq_epi32(ai, _mm256_set1_epi32((int)0x80000000)),
      _mm256_cmpeq_epi32(bi, _mm256_setzero_si256())));
  return _mm256_and_ps(_mm256_or_ps(_mm256_cmp_ps(a, b, _CMP_LT_OS), neg_zero),
                       _mm256_set1_ps(1.0f));
}

/* One operation on all lanes: AVX for the MAC arithmetic, the scalar
 * function lane by lane for the rest. */
static void operate8(Nv2aVshOpcode opcode, const Nv2aVshReg8 in[3],
                     Nv2aVshReg8 *r) {
  int k;
  switch (opcode) {
    case NV2AOP_MOV:
      *r = in[0];
      return;
    case NV2AOP_MUL:
      for (k = 0; k < 4; ++k) r->c[k] = mul8(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_ADD:
      for (k = 0; k < 4; ++k) r->c[k] = _mm256_add_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_MAD:
      for (k = 0; k < 4; ++k) {
        r->c[k] = _mm256_add_ps(mul8(in[0].c[k], in[1].c[k]), in[2].c[k]);
      }
      return;
    case NV2AOP_DP3:
    case NV2AOP_DPH:
    case NV2AOP_DP4: {
      __m256 s = _mm256_add_ps(mul8(in[0].c[0], in[1].c[0]),
                               mul8(in[0].c[1], in[1].c[1]));
      s = _mm256_add_ps(s, mul8(in[0].c[2], in[1].c[2]));
      if (opcode == NV2AOP_DPH) {
        s = _mm256_add_ps(s, in[1].c[3]);
      } else if (opcode == NV2AOP_DP4) {
        s = _mm256_add_ps(s, mul8(in[0].c[3], in[1].c[3]));
      }
      s = fix_inf8(s);
      r->c[0] = r->c[1] = r->c[2] = r->c[3] = s;
      return;
    }
    case NV2AOP_DST:
      r->c[0] = _mm256_set1_ps(1.0f);
      r->c[1] = mul8(in[0].c[1], in[1].c[1]);
      r->c[2] = in[0].c[2];
      r->c[3] = in[1].c[3];
      return;
    case NV2AOP_MIN:
      /* a < b ? a : b */
      for (k = 0; k < 4; ++k) r->c[k] = _mm256_min_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_MAX:
      /* a > b ? a : b */
      for (k = 0; k < 4; ++k) r->c[k] = _mm256_max_ps(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_SLT:
      for (k = 0; k < 4; ++k) r->c[k] = less8(in[0].c[k], in[1].c[k]);
      return;
    case NV2AOP_SGE:
      for (k = 0; k < 4; ++k) {
        r->c[k] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f),
                                              less8(in[0].c[k], in[1].c[k])),
                                _mm256_set1_ps(1.0f));
      }
      return;
    default: {
      float lanes_in[3][4][8], lanes_out[4][8];
      int j, lane;
      for (j = 0; j < 3; ++j) {
        for (k = 0; k < 4; ++k) _mm256_storeu_ps(lanes_in[j][k], in[j].c[k]);
      }
      for (lane = 0; lane < 8; ++lane) {
        float inputs[12], out[4];
        for (j = 0; j < 3; ++j) {
          for (k = 0; k < 4; ++k) inputs[j * 4 + k] = lanes_in[j][k][lane];
        }
        kScalar[opcode](out, inputs);
        for (k = 0; k < 4; ++k) lanes_out[k][lane] = out[k];
      }
      for (k = 0; k < 4; ++k) r->c[k] = _mm256_loadu_ps(lanes_out[k]);
      return;
    }
  }
}

/* apply_operation's writes. */
static void write8(Nv2aVshState8 *state, const Nv2aVshOperation *op,
                   const Nv2aVshReg8 *result) {
  const Nv2aVshOutput *out = op->outputs;
  int i;
  for (i = 0; i < 2; ++i, ++out) {
    Nv2aVshReg8 *reg;
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
      case NV2ART_ADDRESS:
        /* ARL writes the scalar address register without a mask. */
        _mm256_storeu_ps(state->address, result->c[0]);
        continue;
      default: /* none, an input, or a context register (not allowed) */
        continue;
    }
    if (out->writemask & NV2AWM_X) reg->c[0] = result->c[0];
    if (out->writemask & NV2AWM_Y) reg->c[1] = result->c[1];
    if (out->writemask & NV2AWM_Z) reg->c[2] = result->c[2];
    if (out->writemask & NV2AWM_W) reg->c[3] = result->c[3];
  }
}

void nv2a_vsh_emu8_execute(Nv2aVshState8 *state, const Nv2aVshProgram *program) {
  const Nv2aVshStep *step = program->steps;
  for (;;) {
    Nv2aVshReg8 mac_in[3], ilu_in[3], result;
    int i;

    memset(mac_in, 0, sizeof(mac_in));
    memset(ilu_in, 0, sizeof(ilu_in));
    /* Both operations' inputs first, as prepare_inputs does. */
    if (step->mac.opcode) {
      for (i = 0; i < 3; ++i) {
        if (step->mac.inputs[i].type == NV2ART_NONE) {
          break;
        }
        fetch8(state, &step->mac.inputs[i], &mac_in[i]);
      }
    }
    if (step->ilu.opcode) {
      fetch8(state, &step->ilu.inputs[0], &ilu_in[0]);
    }
    if (step->mac.opcode) {
      operate8(step->mac.opcode, mac_in, &result);
      write8(state, &step->mac, &result);
    }
    if (step->ilu.opcode) {
      operate8(step->ilu.opcode, ilu_in, &result);
      write8(state, &step->ilu, &result);
    }
    if (step->is_final) {
      break;
    }
    ++step;
  }
}

#endif  /* NV2A_VSH_HAVE_EMU8 */
