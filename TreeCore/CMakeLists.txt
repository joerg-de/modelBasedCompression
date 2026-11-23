cmake_minimum_required(VERSION 3.10)
project(TreeCore)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory(Main)
add_subdirectory(TestProgramm)
add_subdirectory(PyBridge)