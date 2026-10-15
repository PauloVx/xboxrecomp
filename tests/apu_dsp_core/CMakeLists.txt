cmake_minimum_required(VERSION 3.20)
project(apu_dsp_core_test C)
set(CMAKE_C_STANDARD 11)
add_subdirectory(../.. runtime)
add_executable(apu_dsp_core_test test_main.c test_dma.c)
target_include_directories(apu_dsp_core_test PRIVATE
    ../../src/apu
    ../../src
)
target_link_libraries(apu_dsp_core_test PRIVATE xbox_apu)
enable_testing()
add_test(NAME apu_dsp_core COMMAND apu_dsp_core_test)
