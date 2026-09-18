cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(simd CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

# --- SIMD instruction set ----------------------------------------------------
# The vector backend in include/simd.hpp is x86_64-only today; see ARCH.md and
# ARCH.md. These flags are selected per architecture so that configuring
# on AArch64 succeeds and the build then fails on the missing <immintrin.h> --
# the real blocker -- rather than on an unrecognised -mavx2, which says nothing
# useful. The flags are probed rather than assumed: Apple clang on arm64 rejects
# -march=native and wants -mcpu=native instead.
set(SIMD_ARCH_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
	check_cxx_compiler_flag("-march=native" SIMD_HAVE_MARCH_NATIVE)
	if(SIMD_HAVE_MARCH_NATIVE)
		list(APPEND SIMD_ARCH_FLAGS -march=native)
	endif()
	check_cxx_compiler_flag("-mavx2" SIMD_HAVE_MAVX2)
	if(SIMD_HAVE_MAVX2)
		list(APPEND SIMD_ARCH_FLAGS -mavx2)
	else()
		message(WARNING
			"The compiler does not accept -mavx2. include/simd.hpp needs AVX2 "
			"and will not build."
		)
	endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
	check_cxx_compiler_flag("-march=native" SIMD_HAVE_MARCH_NATIVE)
	if(SIMD_HAVE_MARCH_NATIVE)
		list(APPEND SIMD_ARCH_FLAGS -march=native)
	else()
		check_cxx_compiler_flag("-mcpu=native" SIMD_HAVE_MCPU_NATIVE)
		if(SIMD_HAVE_MCPU_NATIVE)
			list(APPEND SIMD_ARCH_FLAGS -mcpu=native)
		endif()
	endif()
else()
	message(WARNING
		"simd: unrecognised target architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
		"Building without any SIMD instruction-set flags."
	)
endif()
message(STATUS "simd: architecture flags: ${SIMD_ARCH_FLAGS}")

# --- Dependencies ------------------------------------------------------------
# mpfr and gmp were previously linked by bare name, which finds them only when
# they sit in the default link path. Locate them properly so an out-of-tree
# prefix (a conda environment, a local install) can be selected with
# -DCMAKE_PREFIX_PATH=... . Only the code generator and the test need them; the
# simd library itself does not.
# An active conda environment is a common place for these to live, and having
# to pass -DCMAKE_PREFIX_PATH on every fresh build tree is pure friction. Only
# consulted when the variable is set, and an explicit CMAKE_PREFIX_PATH still
# takes precedence because it is searched first.
if(DEFINED ENV{CONDA_PREFIX})
	list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
	message(STATUS "simd: also searching CONDA_PREFIX: $ENV{CONDA_PREFIX}")
endif()

find_path(MPFR_INCLUDE_DIR NAMES mpfr.h)
find_library(MPFR_LIBRARY NAMES mpfr)
find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARY NAMES gmp)

if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY OR NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
	message(FATAL_ERROR
		"simd: mpfr and gmp (headers and libraries) are required by the code "
		"generator and the test.\n"
		"  mpfr.h: ${MPFR_INCLUDE_DIR}\n"
		"  libmpfr: ${MPFR_LIBRARY}\n"
		"  gmp.h: ${GMP_INCLUDE_DIR}\n"
		"  libgmp: ${GMP_LIBRARY}\n"
		"Install them (libmpfr-dev and libgmp-dev on Debian/Ubuntu) or point "
		"CMake at an existing prefix, e.g.\n"
		"  cmake .. -DCMAKE_PREFIX_PATH=/path/to/prefix\n"
		"An activated conda environment (CONDA_PREFIX) is searched "
		"automatically."
	)
endif()
message(STATUS "simd: mpfr: ${MPFR_LIBRARY}")
message(STATUS "simd: gmp:  ${GMP_LIBRARY}")

# Replaces the bare `pthread` link name and the hand-written -pthread flag.
# THREADS_PREFER_PTHREAD_FLAG asks for -pthread over -lpthread where a flag is
# needed at all; on glibc >= 2.34 the pthread symbols live in libc and CMake
# correctly emits neither.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- SIMDe -------------------------------------------------------------------
# Phase 1 of the AArch64 port (ARCH.md): SIMDe reimplements the Intel
# intrinsics over NEON, keeping the AVX2 lane counts so nothing else has to
# change yet. x86_64 uses <immintrin.h> and never sees this.
# The native NEON backend, 4 float / 2 double lanes. Default ON for AArch64:
# it matches the x86 reference on 42 of 48 functions with the six exceptions
# understood, runs about 1.65x faster than the SIMDe route, and needs no
# external dependency, where SIMDe has to be found or fetched. Turn it off with
# -DSIMD_NATIVE_NEON=OFF to fall back to SIMDe.
#
# Note this makes simd_f32::size() 4 rather than 8 by default on AArch64. That
# breaks nobody, AArch64 support being new in the same work, but code written
# against the AVX2 widths needs checking -- see ARCH.md.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
	set(SIMD_NATIVE_NEON_DEFAULT ON)
else()
	set(SIMD_NATIVE_NEON_DEFAULT OFF)
endif()
option(SIMD_NATIVE_NEON "Use the native NEON backend instead of SIMDe"
       ${SIMD_NATIVE_NEON_DEFAULT})
if(SIMD_NATIVE_NEON AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
	message(FATAL_ERROR "simd: SIMD_NATIVE_NEON requires an AArch64 target.")
endif()
if(SIMD_NATIVE_NEON)
	message(STATUS "simd: backend: native NEON (4 float / 2 double lanes)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
	message(STATUS
		"simd: backend: SIMDe over NEON (8 float / 4 double lanes). The native "
		"backend is faster; see ARCH.md."
	)
endif()

set(SIMDE_INCLUDE_DIRS "")
if(NOT SIMD_NATIVE_NEON AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
	find_path(SIMDE_INCLUDE_DIR NAMES simde/x86/avx2.h)
	if(NOT SIMDE_INCLUDE_DIR)
		# Header-only, so populate without adding it as a subproject.
		include(FetchContent)
		FetchContent_Declare(simde
			GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
			GIT_TAG v0.8.2
			GIT_SHALLOW TRUE
		)
		FetchContent_GetProperties(simde)
		if(NOT simde_POPULATED)
			message(STATUS "simd: SIMDe not found locally; fetching v0.8.2")
			FetchContent_Populate(simde)
		endif()
		set(SIMDE_INCLUDE_DIR "${simde_SOURCE_DIR}"
			CACHE PATH "SIMDe include directory" FORCE)
	endif()
	if(NOT SIMDE_INCLUDE_DIR)
		message(FATAL_ERROR
			"simd: SIMDe is required on non-x86 targets and could not be found "
			"or fetched. Install it, or point CMake at a checkout:\n"
			"  cmake .. -DSIMDE_INCLUDE_DIR=/path/to/simde"
		)
	endif()
	set(SIMDE_INCLUDE_DIRS ${SIMDE_INCLUDE_DIR})
	message(STATUS "simd: SIMDe: ${SIMDE_INCLUDE_DIR}")
endif()

set(SIMD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_code")
set(SIMD_GENERATED_MATH "${SIMD_GENERATED_DIR}/src/math.cpp")

# The code generator is an ordinary target rather than something compiled and
# run by execute_process() while configuring. This keeps `cmake` quiet and
# fast, lets the generator be rebuilt with the project's toolchain, and makes
# regeneration a build step that can be re-run when its inputs change.
#
# It is deliberately built WITHOUT the SIMD architecture flags. It includes
# simd.hpp but never calls the vector operations, so the unused inline
# functions are discarded and no AVX2 support is needed. Adding -march=native
# here would enable FMA contraction in the generator's own double-precision
# arithmetic and could shift the coefficients it emits.
add_executable(simd_codegen ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp)
target_compile_definitions(
	simd_codegen PRIVATE SIMD_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(simd_codegen PRIVATE -O3)
if(SIMD_NATIVE_NEON)
	target_compile_definitions(simd_codegen PRIVATE SIMD_NATIVE_NEON)
endif()
target_include_directories(
	simd_codegen PRIVATE ${MPFR_INCLUDE_DIR} ${GMP_INCLUDE_DIR} ${SIMDE_INCLUDE_DIRS}
)
target_link_libraries(simd_codegen PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY} Threads::Threads)

# Regenerated whenever the generator or the sources it splices in change.
add_custom_command(
	OUTPUT ${SIMD_GENERATED_MATH}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SIMD_GENERATED_DIR}/src
	COMMAND simd_codegen
	DEPENDS simd_codegen
	        ${CMAKE_CURRENT_SOURCE_DIR}/include/code.hpp
	        ${CMAKE_CURRENT_SOURCE_DIR}/include/code64.hpp
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating SIMD math functions"
	VERBATIM
)

add_library(simd ${SIMD_GENERATED_MATH})
add_library(simd::simd ALIAS simd)
target_include_directories(
	simd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${SIMDE_INCLUDE_DIRS}
)
#
# -ffp-contract=fast is load-bearing, not a default worth inheriting silently.
# The generated code relies on the compiler contracting a*b+c into an FMA for
# its accuracy. Measured on AArch64 with -ffp-contract=off: f32 exp goes from
# 4 to 68 ULP, f32 erfc from 9 to 71, f32 cosh from 2 to 10, f32 sinh from 3 to
# 9. It is GCC's default, so pinning it changes nothing here, but Clang and
# some distributions default differently and would quietly lose that accuracy.
target_compile_options(simd PUBLIC ${SIMD_ARCH_FLAGS} -ffp-contract=fast)
if(SIMD_NATIVE_NEON)
	target_compile_definitions(simd PUBLIC SIMD_NATIVE_NEON)
endif()
set_target_properties(simd PROPERTIES PUBLIC_HEADER "include/simd.hpp")

# simd_test inherits the architecture flags and the include directory from the
# simd target, which carries them PUBLIC.
add_executable(simd_test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(simd_test PRIVATE ${MPFR_INCLUDE_DIR} ${GMP_INCLUDE_DIR})
target_link_libraries(
	simd_test PUBLIC simd ${MPFR_LIBRARY} ${GMP_LIBRARY} Threads::Threads
)

# The baseline probes. Built as targets rather than by hand-copied g++ lines,
# so they inherit the simd target's architecture flags, include directories and
# -ffp-contract=fast automatically -- a hand-written compile line silently
# drops those and can make the probes disagree with the library they test.
option(SIMD_BUILD_BASELINE_TOOLS "Build the golden/semantics baseline probes" ON)
if(SIMD_BUILD_BASELINE_TOOLS)
	foreach(probe golden semantics)
		add_executable(${probe} ${CMAKE_CURRENT_SOURCE_DIR}/tools/baseline/${probe}.cpp)
		target_link_libraries(${probe} PRIVATE simd)
	endforeach()
endif()

install(TARGETS simd_test DESTINATION bin)
install(TARGETS simd
        LIBRARY DESTINATION simd
        PUBLIC_HEADER DESTINATION include
)
