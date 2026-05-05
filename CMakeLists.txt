cmake_minimum_required(VERSION 3.18)
project(EigenKache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_executable(cuda_kernel_smoke
    cuda/kernel_smoke.cu
    cuda/attention_landmark_kernel.cu
  )
  target_compile_features(cuda_kernel_smoke PRIVATE cxx_std_17)
  add_test(NAME cuda_kernel_smoke COMMAND cuda_kernel_smoke)
else()
  message(STATUS "CUDA compiler not found. CUDA smoke test target will not be built.")
endif()
