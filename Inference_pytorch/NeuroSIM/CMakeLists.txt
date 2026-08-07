cmake_minimum_required(VERSION 3.20)
project(navcim_neurosim LANGUAGES CXX)

file(GLOB NEUROSIM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(neurosim ${NEUROSIM_SOURCES})
target_compile_features(neurosim PRIVATE cxx_std_17)
target_compile_options(neurosim PRIVATE -O3)

if(APPLE)
  set_target_properties(neurosim PROPERTIES OSX_ARCHITECTURES arm64)
endif()
