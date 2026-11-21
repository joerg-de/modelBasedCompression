find_package(Torch REQUIRED)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

add_executable ( TreeCore_example main.cpp )
target_compile_features(TreeCore_example PRIVATE cxx_std_20)
target_link_libraries(TreeCore_example TreeCore)