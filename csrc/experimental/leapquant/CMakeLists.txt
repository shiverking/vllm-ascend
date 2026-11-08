cmake_minimum_required(VERSION 3.16)
project(leapquant_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION Ascend910B3 CACHE STRING "Target Ascend device")
set(RUN_MODE npu CACHE STRING "Kernel execution mode")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.1.0 CACHE PATH "CANN installation")
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(LEAPQUANT_POWER_ITERS 8 CACHE STRING "Cold-start power iterations per rank (warm starts use zero extra iterations)")
set(LEAPQUANT_STREAM_PREFETCH 1 CACHE STRING "Prefetch next head in the opt-in streaming kernel")
set(LEAPQUANT_OPT_TARGETED_VEC_INIT 1 CACHE STRING "Initialize only live vector padding")
set(LEAPQUANT_OPT_TRIM_COEFF 1 CACHE STRING "Compute only live window coefficients")
set(LEAPQUANT_OPT_TRIM_TILED 1 CACHE STRING "Load only live records in tiled Decode")
set(LEAPQUANT_OPT_OMIT_WEIGHT_ZERO 1 CACHE STRING "Skip unused coefficient initialization")
set(LEAPQUANT_V4_FUSED_RECONSTRUCT_SCALE 1 CACHE STRING "Fuse residual row scale with decay")
set(LEAPQUANT_FMA_ROWS 0 CACHE STRING "Experimental fused row contraction")
set(LEAPQUANT_FMA_COLS 0 CACHE STRING "Experimental fused column contraction")
set(LEAPQUANT_DIRECT_NORM 1 CACHE STRING "Normalize residual in place")
set(LEAPQUANT_HALF_BASE 1 CACHE STRING "Use guarded FP16 residual readout")
set(LEAPQUANT_HALF_GUARD 1 CACHE STRING "Fall back to FP32 for large readout operands")
set(LEAPQUANT_V4_DIRECT_BASE_RESULTS 1 CACHE STRING "Consume paired readout without staging")
set(LEAPQUANT_USE_NEWTON_DIV 1 CACHE STRING "Use range-checked Newton reciprocal with Div fallback")
set(LEAPQUANT_DELAY_OUTPUT_STORE 0 CACHE STRING "Experimental delayed output DMA wait")
set(LEAPQUANT_V4_RESIDUAL_PAIR 0 CACHE STRING "Experimental tiled residual paired readout")
set(LEAPQUANT_DUAL_MATVEC 0 CACHE STRING "Experimental interleaved FP32 paired readout")
if(NOT LEAPQUANT_POWER_ITERS MATCHES "^[1-8]$")
  message(FATAL_ERROR "LEAPQUANT_POWER_ITERS must be an integer in [1,8]")
endif()
foreach(flag
    LEAPQUANT_STREAM_PREFETCH LEAPQUANT_OPT_TARGETED_VEC_INIT
    LEAPQUANT_OPT_TRIM_COEFF LEAPQUANT_OPT_TRIM_TILED
    LEAPQUANT_OPT_OMIT_WEIGHT_ZERO LEAPQUANT_V4_FUSED_RECONSTRUCT_SCALE
    LEAPQUANT_FMA_ROWS LEAPQUANT_FMA_COLS LEAPQUANT_DIRECT_NORM
    LEAPQUANT_HALF_BASE LEAPQUANT_HALF_GUARD LEAPQUANT_V4_DIRECT_BASE_RESULTS
    LEAPQUANT_USE_NEWTON_DIV LEAPQUANT_DELAY_OUTPUT_STORE
    LEAPQUANT_V4_RESIDUAL_PAIR LEAPQUANT_DUAL_MATVEC)
  if(NOT "${${flag}}" MATCHES "^[01]$")
    message(FATAL_ERROR "${flag} must be 0 or 1")
  endif()
endforeach()
# CANN builds host and device objects in separate projects. A generated header
# beside a private source snapshot gives both halves exactly the same setting.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(leapquant_build_config.h.in generated/leapquant_build_config.h @ONLY)
configure_file(leapquant_kernel.cpp generated/leapquant_kernel.cpp COPYONLY)
ascendc_library(leapquant_native SHARED ${CMAKE_CURRENT_BINARY_DIR}/generated/leapquant_kernel.cpp)
