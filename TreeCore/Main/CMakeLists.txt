find_package(Torch REQUIRED)

add_compile_options(/fp:fast)
add_compile_options(/Gw)
add_compile_options(/GL)

set(srcFiles "BigUInt.h" "BigUInt.cpp" "helperFunctions.h" "Hidden.h" "Node.h" "PytorchModel.cpp" "PytorchModel.h" "PytorchModelCache.cpp" "PytorchModelCache.h" "PytorchState.cpp" "PytorchState.h" "Settings.h" "strSort.h" "stackAllocOptimizer.hpp")
add_library(TreeCore STATIC ${srcFiles})

target_compile_features(TreeCore PRIVATE cxx_std_20)
target_link_libraries(TreeCore PUBLIC ${TORCH_LIBRARIES})
target_include_directories(TreeCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})