find_package(Torch REQUIRED)

add_executable(TreeCore_example main.cpp)
target_compile_features(TreeCore_example PRIVATE cxx_std_20)

set_target_properties(TreeCore_example PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

target_link_libraries(
    TreeCore_example
    PRIVATE
        TreeCore
        ${TORCH_LIBRARIES}
)