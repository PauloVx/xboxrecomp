/*
 * Kelvin: hardware encodings the vertex stage and primitive assembly decode.
 *
 * Field values of NV_PGRAPH_CSV0_C/CSV0_D/CSV1 (lights, texgen, fog, skinning,
 * material colour sources) and NV097_SET_BEGIN_END. These are register
 * encodings -- facts about the NV2A -- written out here for this engine; the
 * names follow the conventions used across NV2A emulators so the code reads
 * against their documentation.
 */
#ifndef XBOX_NV2A_KELVIN_ENUMS_H
#define XBOX_NV2A_KELVIN_ENUMS_H

/* Light type, two bits per light in NV_PGRAPH_CSV0_D. */
enum {
    LIGHT_OFF = 0,
    LIGHT_INFINITE = 1,
    LIGHT_LOCAL = 2,
    LIGHT_SPOT = 3,
};

/* Texture coordinate generation, per coordinate in NV_PGRAPH_CSV1_A/B. */
enum {
    TEXGEN_DISABLE = 0,
    TEXGEN_EYE_LINEAR = 1,
    TEXGEN_OBJECT_LINEAR = 2,
    TEXGEN_SPHERE_MAP = 3,
    TEXGEN_NORMAL_MAP = 4,
    TEXGEN_REFLECTION_MAP = 5,
};

/* NV_PGRAPH_CONTROL_3_FOG_MODE. 2 and 6 are not valid encodings. */
enum {
    FOG_MODE_LINEAR = 0,
    FOG_MODE_EXP = 1,
    FOG_MODE_EXP2 = 3,
    FOG_MODE_LINEAR_ABS = 4,
    FOG_MODE_EXP_ABS = 5,
    FOG_MODE_EXP2_ABS = 7,
};

/* NV_PGRAPH_CSV0_D_FOGGENMODE: where the fog coordinate comes from. */
enum {
    FOGGEN_SPEC_ALPHA = 0,
    FOGGEN_RADIAL = 1,
    FOGGEN_PLANAR = 2,
    FOGGEN_ABS_PLANAR = 3,
    FOGGEN_FOG_X = 4,
};

/* NV_PGRAPH_CSV0_D_SKIN: vertex blending weights and matrices. */
enum {
    SKINNING_OFF = 0,
    SKINNING_1WEIGHTS = 1,
    SKINNING_2WEIGHTS2MATRICES = 2,
    SKINNING_2WEIGHTS = 3,
    SKINNING_3WEIGHTS3MATRICES = 4,
    SKINNING_3WEIGHTS = 5,
    SKINNING_4WEIGHTS4MATRICES = 6,
};

/* NV_PGRAPH_CSV0_C emission/ambient/diffuse/specular source. */
enum {
    MATERIAL_COLOR_SRC_MATERIAL = 0,
    MATERIAL_COLOR_SRC_DIFFUSE = 1,
    MATERIAL_COLOR_SRC_SPECULAR = 2,
};

/* NV097_SET_BEGIN_END operations; 0 ends a primitive. */
enum {
    PRIM_TYPE_INVALID = 0,
    PRIM_TYPE_POINTS = 1,
    PRIM_TYPE_LINES = 2,
    PRIM_TYPE_LINE_LOOP = 3,
    PRIM_TYPE_LINE_STRIP = 4,
    PRIM_TYPE_TRIANGLES = 5,
    PRIM_TYPE_TRIANGLE_STRIP = 6,
    PRIM_TYPE_TRIANGLE_FAN = 7,
    PRIM_TYPE_QUADS = 8,
    PRIM_TYPE_QUAD_STRIP = 9,
    PRIM_TYPE_POLYGON = 10,
};

/* A transform program instruction is four 32-bit words. */
#define VSH_TOKEN_SIZE 4

#endif
