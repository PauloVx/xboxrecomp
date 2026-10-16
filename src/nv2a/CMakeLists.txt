# NV2A GPU Emulation Library
# Extracted from xemu (https://github.com/xemu-project/xemu)
#
# Register handlers: PMC, PBUS, PTIMER, PFB, PCRTC, PRAMDAC, PFIFO, PGRAPH
# MMIO interception via VEH (Vectored Exception Handling)
# Push buffer parsing and command dispatch

add_library(xbox_nv2a STATIC
    nv2a_core.c
    nv2a_pgraph_d3d11.c
    nv2a_pb_replay.c
    nv2a_mmio_hook.c
    nv2a_pb_test.c
    # Kelvin: the NV2A 3D engine on the CPU (RECOMP_NV2A_KELVIN=1).
    kelvin/kelvin.c
    kelvin/kelvin_mmio.c
    kelvin/kelvin_pgraph.c
    kelvin/kelvin_render.c
    kelvin/kelvin_gpu.c
    kelvin/kelvin_rt.c
    kelvin/kelvin_vsh.c
    kelvin/kelvin_pixel.c
    kelvin/kelvin_texture.c
    kelvin/kelvin_workers.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_cpu.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_disassembler.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_emulator.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_emulator4.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_emulator8.c
    kelvin/nv2a_vsh_cpu/nv2a_vsh_emulator_execution_state.c
)

# The production renderer takes its raster size from the video settings (src/video/pc_video.c); the unit tests that
# include kelvin_render.c directly keep the environment-only behaviour.
set_source_files_properties(kelvin/kelvin_render.c PROPERTIES COMPILE_DEFINITIONS KRES_PC_VIDEO=1)

# Ray tracing shaders (Direct3D 12, shader model 6.5, kelvin/rt): compiled with the Windows SDK's dxc into headers.
# Without dxc the renderer builds without them and ray tracing stays off (kelvin_rt.c).
if(WIN32)
    set(_krt_pf86 "ProgramFiles(x86)")
    file(GLOB _krt_sdk_bins "$ENV{${_krt_pf86}}/Windows Kits/10/bin/10.*/x64")
    list(SORT _krt_sdk_bins COMPARE NATURAL ORDER DESCENDING)
    find_program(KRT_DXC dxc HINTS ${_krt_sdk_bins})
    if(KRT_DXC)
        set(_krt_out ${CMAKE_CURRENT_BINARY_DIR}/krt_shaders)
        set(_krt_headers)
        foreach(_krt_name krt_unpack krt_view krt_reflect krt_calibrate krt_calib_update krt_light krt_reflect_filter krt_light_filter)
            set(_krt_h ${_krt_out}/${_krt_name}.h)
            add_custom_command(
                OUTPUT ${_krt_h}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${_krt_out}
                COMMAND ${KRT_DXC} -nologo -T cs_6_5 -E main -O3 -Fh ${_krt_h} -Vn ${_krt_name}_cs
                        ${CMAKE_CURRENT_SOURCE_DIR}/kelvin/rt/${_krt_name}.hlsl
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kelvin/rt/${_krt_name}.hlsl
                        ${CMAKE_CURRENT_SOURCE_DIR}/kelvin/rt/krt_common.hlsli
                VERBATIM)
            list(APPEND _krt_headers ${_krt_h})
        endforeach()
        target_sources(xbox_nv2a PRIVATE ${_krt_headers})
        target_include_directories(xbox_nv2a PRIVATE ${_krt_out})
        target_compile_definitions(xbox_nv2a PRIVATE KRT_HAVE_SHADERS=1)
    else()
        message(STATUS "dxc not found: the renderer is built without ray tracing")
    endif()
endif()

target_include_directories(xbox_nv2a PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xbox_nv2a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(MSVC)
    target_compile_options(xbox_nv2a PRIVATE
        /W3
        /wd4244
        /wd4146
        /wd4267
        # The CPU renderer's hot loops: AVX2 code generation (no FMA
        # contraction under /fp:precise, so results are unchanged) and no
        # stack-cookie checks in its per-pixel functions.
        /arch:AVX2
        /GS-
    )
    target_compile_definitions(xbox_nv2a PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
else()
    target_compile_options(xbox_nv2a PRIVATE -Wall -Wextra -Wno-unused-parameter -O2)
endif()

target_link_libraries(xbox_nv2a PRIVATE platform)
if(WIN32)
    target_link_libraries(xbox_nv2a PRIVATE kernel32)
    target_link_libraries(xbox_nv2a PRIVATE windowscodecs ole32)  # photo captures as PNG (WIC)
endif()
