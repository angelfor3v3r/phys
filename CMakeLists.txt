cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(phys LANGUAGES CXX)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PHYS_IS_WINDOWS ON)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PHYS_IS_LINUX ON)
else ()
    message(FATAL_ERROR "${PROJECT_NAME}: Unsupported operating system. Only Windows and Linux are supported.")
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(PHYS_IS_CLANG ON)
elseif (PHYS_IS_WINDOWS AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(PHYS_IS_MSVC ON)
else ()
    message(FATAL_ERROR "${PROJECT_NAME}: Unsupported compiler. Only Clang and MSVC are supported.")
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (PHYS_IS_WINDOWS)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmake")

include(get_cpm)

find_package(Threads REQUIRED)

CPMAddPackage(URI "gh:ptsouchlos/thread-pool#0.7.0"
    OPTIONS
    "TP_BUILD_TESTS OFF"
    "TP_BUILD_EXAMPLES OFF"
    "TP_BUILD_BENCHMARKS OFF")
CPMAddPackage(URI "gh:freetype/freetype#VER-2-14-1"
    OPTIONS
    "FT_DISABLE_ZLIB ON"
    "FT_DISABLE_BZIP2 ON"
    "FT_DISABLE_PNG ON"
    "FT_DISABLE_HARFBUZZ ON"
    "FT_DISABLE_BROTLI ON")
CPMAddPackage(URI "gh:libsdl-org/SDL#release-3.4.2"
    OPTIONS
    "SDL_SHARED OFF"
    "SDL_STATIC ON"
    "SDL_AUDIO OFF"
    "SDL_CAMERA OFF"
    "SDL_DIALOG OFF"
    "SDL_GPU OFF"
    "SDL_HAPTIC OFF"
    "SDL_HIDAPI OFF"
    "SDL_JOYSTICK OFF"
    "SDL_POWER OFF"
    "SDL_SENSOR OFF"
    "SDL_TRAY OFF"
    "SDL_DUMMYVIDEO OFF"
    "SDL_OFFSCREEN OFF"
    "SDL_OPENGLES OFF"
    "SDL_RENDER_GPU OFF")
CPMAddPackage("gh:ocornut/imgui@1.92.6-docking")
CPMAddPackage(URI "gh:erincatto/box2d@3.1.1" OPTIONS "BOX2D_AVX2 ON")

if (imgui_ADDED)
    file(GLOB IMGUI_HEADERS
        ${imgui_SOURCE_DIR}/*.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.h
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.h
        ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.h)
    list(REMOVE_ITEM IMGUI_HEADERS ${imgui_SOURCE_DIR}/imconfig.h ${imgui_SOURCE_DIR}/imstb_truetype.h)
    file(GLOB IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/*.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp)
    list(REMOVE_ITEM IMGUI_SOURCES ${imgui_SOURCE_DIR}/imgui_demo.cpp)
    add_library(imgui STATIC ${IMGUI_HEADERS} ${IMGUI_SOURCES})
    target_compile_definitions(imgui PUBLIC IMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/imconfig.hpp")
    target_include_directories(imgui
        PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${imgui_SOURCE_DIR}/misc/cpp
        ${imgui_SOURCE_DIR}/misc/freetype)
    target_link_libraries(imgui PUBLIC freetype SDL3::SDL3-static)
    add_library(imgui::imgui ALIAS imgui)
endif ()


# set(PHYS_HEADERS ...)
set(PHYS_SOURCES src/main.cpp)
# set(PHYS_RESOURCES ...)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PHYS_SOURCES})
add_executable(phys WIN32 ${PHYS_SOURCES})
target_compile_features(phys PRIVATE cxx_std_23)
target_compile_definitions(phys PRIVATE
    $<IF:$<CONFIG:Debug>,PHYS_DEBUG=1,PHYS_DEBUG=0>
    $<$<BOOL:${PHYS_IS_WINDOWS}>:NOMINMAX UNICODE _UNICODE WINVER=0x0A00 _WIN32_WINNT=0x0A00>
    SDL_MAIN_USE_CALLBACKS)
target_include_directories(phys PRIVATE src)

if (PHYS_IS_CLANG)
    target_compile_options(phys
        PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -pedantic
        -Werror=unused-variable
        -Wno-non-virtual-dtor
        -Wno-unused-parameter
        -Wno-unused-private-field
        -Wno-cast-function-type
        -Wno-missing-designated-field-initializers
        -Wno-language-extension-token)

    target_compile_options(phys PRIVATE $<$<CONFIG:Debug>:-g3 -O0>)
    target_compile_options(phys PRIVATE $<$<CONFIG:Release>:-fno-ident -g0 -O3>)
    target_compile_options(phys PRIVATE $<$<CONFIG:RelWithDebInfo>:-g3 -O3>)

    if (PHYS_IS_LINUX)
        target_link_options(phys PRIVATE -static-libgcc -static-libstdc++)
    endif ()
elseif (PHYS_IS_MSVC)
    target_compile_options(phys
        PRIVATE
        /W4)

    # TODO: Debug/Release flags.
endif ()

target_link_libraries(phys PRIVATE dp::thread-pool SDL3::SDL3-static imgui::imgui box2d::box2d $<$<BOOL:${PHYS_IS_WINDOWS}>:dwmapi>)
