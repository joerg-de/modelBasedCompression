find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

set(SOURCES
    PythonBridgeMain.cpp
    PythonBridgePytorchModel.cpp
    PythonBridgePytorchState.cpp
    PythonCallBackThrottler.cpp
)
set(HEADERS
    PythonBridgeMain.h
    PythonBridgePytorchModel.h
    PythonBridgePytorchState.h
    PythonCallBackThrottler.h
)


add_library(TreeCorePyBridge MODULE ${SOURCES})
target_compile_features(TreeCorePyBridge PRIVATE cxx_std_20)
target_sources(TreeCorePyBridge PRIVATE ${HEADERS})

set_target_properties(TreeCorePyBridge PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(
    TreeCorePyBridge
    PRIVATE
        TreeCore
        Python::Python
)