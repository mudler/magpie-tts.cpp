cmake_minimum_required(VERSION 3.16)
project(magpie-tts-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(MAGPIE_BUILD_CLI     "Build the magpie-cli example"       ON)
option(MAGPIE_BUILD_TESTS   "Build tests"                        ON)
option(MAGPIE_SHARED        "Build magpie-tts as a shared lib"   OFF)
option(MAGPIE_GGML_CUDA     "Enable ggml CUDA backend"           OFF)
option(MAGPIE_GGML_METAL    "Enable ggml Metal backend"          OFF)
option(MAGPIE_GGML_VULKAN   "Enable ggml Vulkan backend"         OFF)
option(MAGPIE_GGML_HIP      "Enable ggml HIP backend"            OFF)

# The magpie GGUF keeps NeMo state-dict tensor names verbatim; the longest
# (codec res-block snake alphas) is 102 chars. ggml's default GGML_MAX_NAME=64
# makes gguf_init_from_file REJECT such files outright, so raise it project-wide.
# It sizes ggml_tensor::name, so it must be identical in ggml and in every
# consumer TU -- hence a directory-scope definition ahead of add_subdirectory.
add_compile_definitions(GGML_MAX_NAME=128)

# Forward backend toggles to ggml
set(GGML_CUDA   ${MAGPIE_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL  ${MAGPIE_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN ${MAGPIE_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP    ${MAGPIE_GGML_HIP}    CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
endif()
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

set(MAGPIE_SOURCES
    src/common.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/encoder.cpp
    src/decoder.cpp
    src/local_transformer.cpp
    src/prior.cpp
    src/codec.cpp
    src/audio_io.cpp
    src/magpie_tts.cpp
    src/magpie_tts_capi.cpp
)

if(MAGPIE_SHARED)
    add_library(magpie-tts SHARED ${MAGPIE_SOURCES})
    set_target_properties(magpie-tts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    add_library(magpie-tts STATIC ${MAGPIE_SOURCES})
endif()

target_include_directories(magpie-tts PUBLIC include)
target_include_directories(magpie-tts PRIVATE src third_party)
target_link_libraries(magpie-tts PUBLIC ggml)
target_compile_options(magpie-tts PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Statically link enabled ggml backends (LocalAI purego consumers need one .so)
foreach(_be cuda metal vulkan hip blas)
    string(TOUPPER ${_be} _BE)
    if(TARGET ggml-${_be})
        target_link_libraries(magpie-tts PUBLIC ggml-${_be})
        target_compile_definitions(magpie-tts PRIVATE MAGPIE_HAVE_${_BE})
    endif()
endforeach()

find_package(Threads REQUIRED)
target_link_libraries(magpie-tts PUBLIC Threads::Threads)

if(MAGPIE_BUILD_CLI)
    add_subdirectory(examples/cli)
endif()

if(MAGPIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
