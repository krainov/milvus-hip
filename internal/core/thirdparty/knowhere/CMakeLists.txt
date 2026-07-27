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
# AMD/HIP builds can override the upstream Knowhere source with:
#   -DMILVUS_KNOWHERE_GIT_REPOSITORY=... -DMILVUS_KNOWHERE_VERSION=...
#   -DMILVUS_KNOWHERE_SOURCE_DIR=/path/to/local/knowhere  (skip git fetch)
if(NOT DEFINED MILVUS_KNOWHERE_VERSION)
  set(MILVUS_KNOWHERE_VERSION "v2.5.2")
endif()
if(NOT DEFINED MILVUS_KNOWHERE_GIT_REPOSITORY)
  set(MILVUS_KNOWHERE_GIT_REPOSITORY "https://github.com/zilliztech/knowhere.git")
endif()
set(MILVUS_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE MILVUS_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${MILVUS_GPU_BACKEND}" MILVUS_GPU_BACKEND)
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

if ( MILVUS_GPU_VERSION STREQUAL "ON" )
  if(MILVUS_GPU_BACKEND STREQUAL "HIP")
    set(WITH_CUVS ON CACHE BOOL "" FORCE )
    set(WITH_HIP ON CACHE BOOL "" FORCE )
    set(WITH_RAFT OFF CACHE BOOL "" FORCE )
    message(STATUS "Milvus GPU backend: HIP (hipVS/hipRAFT)")
  elseif(MILVUS_GPU_BACKEND STREQUAL "CUDA")
    set(WITH_CUVS OFF CACHE BOOL "" FORCE )
    set(WITH_HIP OFF CACHE BOOL "" FORCE )
    set(WITH_RAFT ON CACHE BOOL "" FORCE )
    message(STATUS "Milvus GPU backend: CUDA (RAFT)")
  else()
    message(FATAL_ERROR "Unsupported MILVUS_GPU_BACKEND=${MILVUS_GPU_BACKEND}")
  endif()
endif ()

if(MILVUS_GPU_BACKEND STREQUAL "HIP")
  # Prefer hipVS/hipRAFT + ROCm ahead of Conan boost for HIP package discovery.
  # core_build.sh overwrites INSTALL_PREFIX, so use the dedicated HIP variables.
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
  foreach(_xx_cand
      "${_milvus_hip_prefix}/lib/cmake/xxHash"
      "${_milvus_hip_prefix}/lib64/cmake/xxHash")
    if(EXISTS "${_xx_cand}/xxHashConfig.cmake" OR EXISTS "${_xx_cand}/xxhash-config.cmake")
      set(xxHash_DIR "${_xx_cand}" CACHE PATH "xxHash CMake package dir" FORCE)
      break()
    endif()
  endforeach()
  message(STATUS "Milvus HIP CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
  message(STATUS "Milvus HIP xxHash_DIR: ${xxHash_DIR}")
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
