#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

# Update KNOWHERE_VERSION for the first occurrence
milvus_add_pkg_config("knowhere")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
# Layer 3 AMD/HIP: default to DXC Knowhere 2.5 (HIP merged). Override with:
#   -DMILVUS_KNOWHERE_GIT_REPOSITORY=... -DMILVUS_KNOWHERE_VERSION=...
#   -DMILVUS_KNOWHERE_SOURCE_DIR=/path/to/local/knowhere  (skip git fetch)
if(NOT DEFINED MILVUS_KNOWHERE_VERSION)
  set(MILVUS_KNOWHERE_VERSION "2.5")
endif()
if(NOT DEFINED MILVUS_KNOWHERE_GIT_REPOSITORY)
  set(MILVUS_KNOWHERE_GIT_REPOSITORY "https://github.dxc.com/llmkb-internal/knowhere.git")
endif()
set( KNOWHERE_VERSION ${MILVUS_KNOWHERE_VERSION} )
set( GIT_REPOSITORY ${MILVUS_KNOWHERE_GIT_REPOSITORY} )
message(STATUS "Knowhere repo: ${GIT_REPOSITORY}")
message(STATUS "Knowhere version: ${KNOWHERE_VERSION}")
if(DEFINED MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "Knowhere local source: ${MILVUS_KNOWHERE_SOURCE_DIR}")
endif()

message(STATUS "Building knowhere-${KNOWHERE_SOURCE_VER} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

if ( BUILD_DISK_ANN STREQUAL "ON" )
    set(WITH_DISKANN ON CACHE BOOL "" FORCE )
else ()
    set(WITH_DISKANN OFF CACHE BOOL "" FORCE )
endif ()

if ( INDEX_ENGINE STREQUAL "cardinal" )
    set(WITH_CARDINAL ON CACHE BOOL "" FORCE )
else ()
    set(WITH_CARDINAL OFF CACHE BOOL "" FORCE )
endif()

if ( MILVUS_GPU_VERSION )
  # GPU Knowhere uses cuVS on both backends. HIP is an explicit AMD-only opt-in.
  set(WITH_CUVS ON CACHE BOOL "" FORCE )
  set(_milvus_want_hip OFF)
  if(DEFINED MILVUS_WITH_HIP AND MILVUS_WITH_HIP)
    set(_milvus_want_hip ON)
  elseif(DEFINED ENV{MILVUS_WITH_HIP})
    string(TOUPPER "$ENV{MILVUS_WITH_HIP}" _milvus_hip_env)
    if(_milvus_hip_env STREQUAL "1"
       OR _milvus_hip_env STREQUAL "ON"
       OR _milvus_hip_env STREQUAL "TRUE"
       OR _milvus_hip_env STREQUAL "YES")
      set(_milvus_want_hip ON)
    endif()
    unset(_milvus_hip_env)
  elseif(DEFINED ENV{KNOWHERE_WITH_HIP})
    string(TOUPPER "$ENV{KNOWHERE_WITH_HIP}" _milvus_hip_env)
    if(_milvus_hip_env STREQUAL "1"
       OR _milvus_hip_env STREQUAL "ON"
       OR _milvus_hip_env STREQUAL "TRUE"
       OR _milvus_hip_env STREQUAL "YES")
      set(_milvus_want_hip ON)
    endif()
    unset(_milvus_hip_env)
  endif()
  set(WITH_HIP ${_milvus_want_hip} CACHE BOOL
      "Link Knowhere against hipVS/hipRAFT (ROCm); OFF for CUDA" FORCE)
  if(_milvus_want_hip)
    set(WITH_RAFT OFF CACHE BOOL "" FORCE )
    message(STATUS "MILVUS Layer3 HIP: WITH_CUVS=ON WITH_HIP=ON (hipVS/hipRAFT)")
  else()
    message(STATUS "MILVUS GPU: WITH_CUVS=ON WITH_HIP=OFF (CUDA)")
  endif()
  unset(_milvus_want_hip)
endif ()

if(WITH_HIP)
# Prefer hipVS/hipRAFT + ROCm ahead of Conan boost for find_package(cuvs/raft/hip).
# Do NOT read ENV{INSTALL_PREFIX}: Milvus core_build.sh overwrites it to the Milvus
# output tree, which hides hipVS/xxHash under ~/rocmds_check_gfx1100/install.
# Do NOT prepend Knowhere Conan CMakeDeps generators - their xxHashConfig.cmake
# calls check_build_type_defined and breaks configure outside conan toolchain.
set(_milvus_hip_prefix "$ENV{MILVUS_HIP_INSTALL_PREFIX}")
if(_milvus_hip_prefix STREQUAL "")
  set(_milvus_hip_prefix "$ENV{ROCMDS_INSTALL_PREFIX}")
endif()
if(_milvus_hip_prefix STREQUAL "")
  set(_milvus_hip_prefix "$ENV{HOME}/rocmds_check_gfx1100/install")
endif()
set(_milvus_rocm "$ENV{ROCM_PATH}")
if(_milvus_rocm STREQUAL "")
  set(_milvus_rocm "/opt/rocm")
endif()
set(CMAKE_PREFIX_PATH "${_milvus_hip_prefix};${_milvus_rocm};${CONAN_BOOST_ROOT};${CMAKE_PREFIX_PATH}")
# Standalone xxHash only (installed under hip prefix), never Conan generators.
foreach(_xx_cand
    "${_milvus_hip_prefix}/lib/cmake/xxHash"
    "${_milvus_hip_prefix}/lib64/cmake/xxHash")
  if(EXISTS "${_xx_cand}/xxHashConfig.cmake" OR EXISTS "${_xx_cand}/xxhash-config.cmake")
    set(xxHash_DIR "${_xx_cand}" CACHE PATH "xxHash CMake package dir" FORCE)
    break()
  endif()
endforeach()
message(STATUS "MILVUS Layer3 Knowhere CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "MILVUS Layer3 Knowhere xxHash_DIR: ${xxHash_DIR}")
else()
  set(CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT})
endif()

if(DEFINED MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")
  set(knowhere_SOURCE_DIR "${MILVUS_KNOWHERE_SOURCE_DIR}")
  set(knowhere_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/knowhere-build")
  file(MAKE_DIRECTORY "${knowhere_BINARY_DIR}")
  add_subdirectory( ${knowhere_SOURCE_DIR} ${knowhere_BINARY_DIR} )
else()
  FetchContent_Declare(
        knowhere
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${KNOWHERE_VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

  FetchContent_GetProperties( knowhere )
  if ( NOT knowhere_POPULATED )
    FetchContent_Populate( knowhere )

    # Adding the following target:
    # knowhere
    add_subdirectory( ${knowhere_SOURCE_DIR}
                      ${knowhere_BINARY_DIR} )
  endif()
endif()

# get prometheus COMPILE_OPTIONS
get_property( var DIRECTORY "${knowhere_SOURCE_DIR}" PROPERTY COMPILE_OPTIONS )
message( STATUS "knowhere src compile options: ${var}" )

set( KNOWHERE_INCLUDE_DIR ${knowhere_SOURCE_DIR}/include CACHE INTERNAL "Path to knowhere include directory" )
