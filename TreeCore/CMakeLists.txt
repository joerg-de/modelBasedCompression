project(TreeCore)
cmake_minimum_required(VERSION 3.10.1)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Main)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TestProgramm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/PyBridge)