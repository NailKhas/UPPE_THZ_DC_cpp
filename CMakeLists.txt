cmake_minimum_required(VERSION 3.21)
project(uppe_thz_dc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(UPPE_BACKEND "MKL" CACHE STRING "Numerical backend: MKL or FFTW_OPENBLAS")
set_property(CACHE UPPE_BACKEND PROPERTY STRINGS MKL FFTW_OPENBLAS)
option(UPPE_NATIVE "Optimize for the CPU of the build machine" ON)
option(UPPE_BUILD_TESTS "Build unit tests" ON)
set(UPPE_FOCUS_CM "20.0" CACHE STRING "Focal distance in cm (compile-time constant c::focus)")
set(UPPE_ZSIC_CM "18.0" CACHE STRING "Start distance of the propagation in cm (compile-time constant c::zsic)")
set(UPPE_PRELENS_CM "400.0" CACHE STRING "Free path of the source beam before the lens in cm, 0 allowed (c::prelens)")
set(UPPE_ENERGY_MJ "0.5" CACHE STRING "Pulse energy in mJ (c::WW0)")
set(UPPE_A0_CM "" CACHE STRING "Length unit a0 in cm (c::a0); empty = a00*(1 - zsic/focus) as in the Fortran")
set(UPPE_RMAX "20.0" CACHE STRING "Optical grid radius in units of a0 (c::Rmax)")
set(UPPE_NR_OPT "1000" CACHE STRING "Number of optical radial nodes (default of Sizes::nR_opt)")

# Checks that ${var} is a positive (or, with ALLOW_ZERO, non-negative) number and sets ${out} to it as a
# C++ double literal.
function(uppe_double_literal var out)
  cmake_parse_arguments(PARSE_ARGV 2 arg "ALLOW_ZERO" "" "")
  set(v "${${var}}")
  if(NOT v MATCHES "^[0-9]+(\\.[0-9]*)?$" OR (NOT arg_ALLOW_ZERO AND v MATCHES "^0*(\\.0*)?$"))
    message(FATAL_ERROR "${var} must be a positive number (or 0 where allowed), got '${v}'")
  endif()
  if(NOT v MATCHES "\\.")
    set(v "${v}.")
  endif()
  set(${out} "${v}0" PARENT_SCOPE)
endfunction()
uppe_double_literal(UPPE_FOCUS_CM _uppe_focus_literal)
uppe_double_literal(UPPE_ZSIC_CM _uppe_zsic_literal)
uppe_double_literal(UPPE_PRELENS_CM _uppe_prelens_literal ALLOW_ZERO)
uppe_double_literal(UPPE_ENERGY_MJ _uppe_energy_literal)
uppe_double_literal(UPPE_RMAX _uppe_rmax_literal)
if(NOT UPPE_NR_OPT MATCHES "^[1-9][0-9]*$" OR UPPE_NR_OPT LESS 3)
  message(FATAL_ERROR "UPPE_NR_OPT must be an integer >= 3, got '${UPPE_NR_OPT}'")
endif()
set(_uppe_defs
  UPPE_FOCUS_CM=${_uppe_focus_literal}
  UPPE_ZSIC_CM=${_uppe_zsic_literal}
  UPPE_PRELENS_CM=${_uppe_prelens_literal}
  UPPE_ENERGY_MJ=${_uppe_energy_literal}
  UPPE_RMAX=${_uppe_rmax_literal}
  UPPE_NR_OPT=${UPPE_NR_OPT})
if(NOT UPPE_A0_CM STREQUAL "")
  uppe_double_literal(UPPE_A0_CM _uppe_a0_literal)
  list(APPEND _uppe_defs UPPE_A0_CM=${_uppe_a0_literal})
endif()
message(STATUS "Focal distance: ${UPPE_FOCUS_CM} cm, start distance zsic: ${UPPE_ZSIC_CM} cm, "
               "path before the lens: ${UPPE_PRELENS_CM} cm, energy: ${UPPE_ENERGY_MJ} mJ")
message(STATUS "a0: ${UPPE_A0_CM} cm (empty = a00*(1 - zsic/focus)), Rmax: ${UPPE_RMAX}, nR_opt: ${UPPE_NR_OPT}")

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(uppe_core STATIC
  src/blas.cpp
  src/checkpoint.cpp
  src/fft.cpp
  src/first_step.cpp
  src/grids.cpp
  src/hankel.cpp
  src/nonlinear.cpp
  src/output.cpp
  src/physics.cpp
  src/propagation.cpp
  src/simulation.cpp
)
target_include_directories(uppe_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(uppe_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_definitions(uppe_core PUBLIC ${_uppe_defs})

# The port follows the Fortran operation order and IEEE semantics, so value-changing
# floating-point optimizations (-ffast-math, /fp:fast, icx default -fp-model=fast) must stay off.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(uppe_core PUBLIC /fp:precise)
    if(UPPE_NATIVE)
      target_compile_options(uppe_core PUBLIC /QxHost)
    endif()
  else()
    target_compile_options(uppe_core PUBLIC -fp-model=precise)
    if(UPPE_NATIVE)
      target_compile_options(uppe_core PUBLIC -xHost)
    endif()
  endif()
elseif(MSVC)
  target_compile_options(uppe_core PUBLIC /fp:precise /W3)
  if(UPPE_NATIVE)
    target_compile_options(uppe_core PUBLIC /arch:AVX2)
  endif()
else()
  target_compile_options(uppe_core PUBLIC -fno-fast-math -fno-math-errno -Wall -Wextra)
  if(UPPE_NATIVE)
    target_compile_options(uppe_core PUBLIC -march=native)
  endif()
endif()

if(UPPE_BACKEND STREQUAL "MKL")
  # Our own OpenMP loops call BLAS/FFT from many threads, so MKL itself must be sequential.
  if(NOT DEFINED MKL_THREADING)
    set(MKL_THREADING sequential)
  endif()
  if(NOT DEFINED MKL_INTERFACE)
    set(MKL_INTERFACE lp64)
  endif()
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(uppe_core PUBLIC MKL::MKL)
  target_compile_definitions(uppe_core PUBLIC UPPE_USE_MKL)
elseif(UPPE_BACKEND STREQUAL "FFTW_OPENBLAS")
  find_package(PkgConfig QUIET)

  set(_uppe_fftw_found FALSE)
  find_package(FFTW3 CONFIG QUIET)
  if(TARGET FFTW3::fftw3)
    target_link_libraries(uppe_core PUBLIC FFTW3::fftw3)
    set(_uppe_fftw_found TRUE)
  elseif(PKG_CONFIG_FOUND)
    pkg_check_modules(UPPE_FFTW3 QUIET IMPORTED_TARGET fftw3)
    if(UPPE_FFTW3_FOUND)
      target_link_libraries(uppe_core PUBLIC PkgConfig::UPPE_FFTW3)
      set(_uppe_fftw_found TRUE)
    endif()
  endif()
  if(NOT _uppe_fftw_found)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3-3 fftw3-3)
    if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
      message(FATAL_ERROR "FFTW3 (double precision) not found")
    endif()
    target_include_directories(uppe_core PUBLIC ${FFTW3_INCLUDE_DIR})
    target_link_libraries(uppe_core PUBLIC ${FFTW3_LIBRARY})
  endif()

  set(_uppe_blas_found FALSE)
  find_package(OpenBLAS CONFIG QUIET)
  if(TARGET OpenBLAS::OpenBLAS)
    target_link_libraries(uppe_core PUBLIC OpenBLAS::OpenBLAS)
    set(_uppe_blas_found TRUE)
  elseif(PKG_CONFIG_FOUND)
    pkg_check_modules(UPPE_OPENBLAS QUIET IMPORTED_TARGET openblas)
    if(UPPE_OPENBLAS_FOUND)
      target_link_libraries(uppe_core PUBLIC PkgConfig::UPPE_OPENBLAS)
      set(_uppe_blas_found TRUE)
    endif()
  endif()
  if(NOT _uppe_blas_found)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas)
    if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
      message(FATAL_ERROR "OpenBLAS not found")
    endif()
    target_include_directories(uppe_core PUBLIC ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(uppe_core PUBLIC ${OPENBLAS_LIBRARY})
  endif()
  target_compile_definitions(uppe_core PUBLIC UPPE_USE_OPENBLAS)
else()
  message(FATAL_ERROR "Unknown UPPE_BACKEND '${UPPE_BACKEND}' (use MKL or FFTW_OPENBLAS)")
endif()

add_executable(uppe_thz_dc src/main.cpp)
target_link_libraries(uppe_thz_dc PRIVATE uppe_core)

if(UPPE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
