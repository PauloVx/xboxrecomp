/* Keep a diagnostic record's multiple stdio writes together.
 * Released under LGPL-2.1-or-later, as part of the Kelvin renderer. */
#ifndef XBOX_NV2A_KELVIN_TRACE_H
#define XBOX_NV2A_KELVIN_TRACE_H

#include <stdio.h>

static inline void kelvin_trace_record_begin(FILE *stream)
{
#ifdef _MSC_VER
    _lock_file(stream);
#else
    flockfile(stream);
#endif
}

static inline void kelvin_trace_record_end(FILE *stream)
{
#ifdef _MSC_VER
    _unlock_file(stream);
#else
    funlockfile(stream);
#endif
}

#endif
