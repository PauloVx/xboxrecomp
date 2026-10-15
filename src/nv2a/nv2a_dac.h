#ifndef XBOXRECOMP_NV2A_DAC_H
#define XBOXRECOMP_NV2A_DAC_H

#include <stdint.h>

/* The VGA DAC palette in PRMDIO (offsets within that block, one byte each). On NV2A it is the display's colour
 * lookup, which the XDK's gamma ramp fills: 0x3C8 selects the entry, then three 0x3C9 writes give its red, green
 * and blue. Each finished entry goes to the presenter. */
void nv2a_dac_write(uint32_t offset, uint8_t value);
uint8_t nv2a_dac_read(uint32_t offset);

#endif
