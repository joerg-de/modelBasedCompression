find_package(Torch REQUIRED)

set(SOURCES
    BigUInt.cpp
    PytorchModel.cpp
    PytorchModelCache.cpp
    PytorchState.cpp
)
set(HEADERS
    BigUInt.h
    helperFunctions.h
    Hidden.h
    Node.h
    PytorchModel.h
    PytorchModelCache.h
    PytorchState.h
    Settings.h
    strSort.h
    stackAllocOptimizer.hpp
)

add_library(TreeCore STATIC ${SOURCES})
target_compile_features(TreeCore PRIVATE cxx_std_20)
target_sources(TreeCore PRIVATE ${HEADERS})

if(MSVC)
    target_compile_options(TreeCore PRIVATE /fp:fast /Gw /GL)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(TreeCore PRIVATE -O3 -ffast-math -flto)
endif()

set_target_properties(TreeCore PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

target_link_libraries(TreeCore PUBLIC ${TORCH_LIBRARIES})
target_include_directories(TreeCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
