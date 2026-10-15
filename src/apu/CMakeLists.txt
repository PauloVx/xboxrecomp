# MCPX APU Emulation Library
# Extracted from xemu (https://github.com/xemu-project/xemu)
#
# Voice Processor (VP): ADPCM/PCM decode, sample mixing, envelopes, filters
# Global/Encode Processors (GP/EP): DSP56300 execution and MCPX DMA

add_library(xbox_apu STATIC
    apu_core.c
    apu_vp.c
    apu_dsp.c
    apu_dma.c
    apu_dsp_engine.c
    apu_mmio_hook.c
    apu_xaudio2.c
)

target_include_directories(xbox_apu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# DSP56300 is a separately maintained MIT-licensed Rust core with a C ABI.
# Its source is the third_party/dsp56300 submodule of the game project; build the static library into
# this CMake tree so the runtime and standalone APU tests use the same engine.
get_filename_component(BLACK_PROJECT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../" ABSOLUTE)
set(DSP56300_SOURCE_DIR "${BLACK_PROJECT_ROOT}/third_party/dsp56300" CACHE PATH
    "Pinned DSP56300 source checkout")
if(NOT EXISTS "${DSP56300_SOURCE_DIR}/crates/emu-ffi/Cargo.toml")
    message(FATAL_ERROR
        "Pinned DSP56300 source is missing. Run: git submodule update --init --recursive")
endif()
find_program(CARGO_EXECUTABLE NAMES cargo cargo.exe)
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "Rust/Cargo 1.98 or newer is required to build the DSP56300 core.")
endif()

if(WIN32 AND MSVC)
    set(DSP56300_TARGET_TRIPLE x86_64-pc-windows-msvc)
    set(DSP56300_LIBRARY
        "${CMAKE_CURRENT_BINARY_DIR}/dsp56300/${DSP56300_TARGET_TRIPLE}/release/dsp56300_emu_ffi.lib")
else()
    set(DSP56300_TARGET_TRIPLE x86_64-unknown-linux-gnu)
    set(DSP56300_LIBRARY
        "${CMAKE_CURRENT_BINARY_DIR}/dsp56300/${DSP56300_TARGET_TRIPLE}/release/libdsp56300_emu_ffi.a")
endif()
file(GLOB_RECURSE DSP56300_RUST_SOURCES CONFIGURE_DEPENDS
    "${DSP56300_SOURCE_DIR}/crates/core/*.rs"
    "${DSP56300_SOURCE_DIR}/crates/emu/*.rs"
    "${DSP56300_SOURCE_DIR}/crates/emu-ffi/*.rs"
    "${DSP56300_SOURCE_DIR}/crates/emu-ffi/*.c"
)
add_custom_command(
    OUTPUT "${DSP56300_LIBRARY}"
    COMMAND "${CARGO_EXECUTABLE}" build --release --locked
            --no-default-features -p dsp56300-emu-ffi
            --target "${DSP56300_TARGET_TRIPLE}"
            --target-dir "${CMAKE_CURRENT_BINARY_DIR}/dsp56300"
    WORKING_DIRECTORY "${DSP56300_SOURCE_DIR}"
    DEPENDS
        "${DSP56300_SOURCE_DIR}/Cargo.toml"
        "${DSP56300_SOURCE_DIR}/Cargo.lock"
        ${DSP56300_RUST_SOURCES}
    COMMENT "Building the pinned DSP56300 JIT library"
    VERBATIM
)
add_custom_target(dsp56300_ffi_build DEPENDS "${DSP56300_LIBRARY}")
add_dependencies(xbox_apu dsp56300_ffi_build)
target_include_directories(xbox_apu PUBLIC
    "${DSP56300_SOURCE_DIR}/crates/emu-ffi/include"
)
target_link_libraries(xbox_apu PRIVATE "${DSP56300_LIBRARY}")

# qemu_shim.h lives in nv2a/ - shared QEMU type abstraction
target_include_directories(xbox_apu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nv2a
)

if(MSVC)
    target_compile_options(xbox_apu PRIVATE
        /W3
        /wd4244  # conversion warnings (float/double/int)
        /wd4146  # unary minus on unsigned
        /wd4267  # size_t to int conversion
    )
    target_compile_definitions(xbox_apu PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
    )
endif()

if(WIN32)
    # xaudio2_8 for XAudio2Create, which apu_xaudio2.c calls directly.
    # It was missing: nothing had ever referenced this library from a linked
    # target, so the archive was built and never pulled in, and the first
    # target to actually call into the APU failed at link with an undefined
    # XAudio2Create rather than at build time here.
    target_link_libraries(xbox_apu PRIVATE
        platform kernel32 winmm xaudio2_8 ole32 ntdll ws2_32 userenv
    )
else()
    # Linux: platform layer (Win32 primitives on POSIX) + SDL2 audio + libm
    find_package(SDL2 REQUIRED)
    target_link_libraries(xbox_apu PRIVATE platform SDL2::SDL2 m)
endif()
