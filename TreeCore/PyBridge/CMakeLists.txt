find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(srcFiles "PythonBridgeMain.cpp" "PythonBridgeMain.h" "PythonBridgePytorchModel.cpp" "PythonBridgePytorchModel.h" "PythonBridgePytorchState.cpp" "PythonBridgePytorchState.h" "PythonCallBackThrottler.cpp" "PythonCallBackThrottler.h")
add_library(TreeCorePyBridge SHARED ${srcFiles})
target_compile_features(TreeCorePyBridge PRIVATE cxx_std_20)
target_link_libraries(TreeCorePyBridge PUBLIC TreeCore Python::Python)