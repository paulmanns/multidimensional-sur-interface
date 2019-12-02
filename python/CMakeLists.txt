find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

find_program(CYTHON "cython")

if(NOT CYTHON)
  message(FATAL_ERROR "Cython not found on this system.")
endif()

execute_process(COMMAND
${PYTHON_EXECUTABLE} -c "import sys, numpy;
sys.stdout.write(numpy.get_include())"
OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
RESULT_VARIABLE NUMPY_NOT_FOUND)

if(NUMPY_NOT_FOUND)
message(FATAL_ERROR "NumPy headers not found")
endif()

add_custom_command(
OUTPUT hilbert_cython.cpp
  COMMAND ${CYTHON} -o hilbert_cython.cpp ${CMAKE_CURRENT_SOURCE_DIR}/hilbert_cython.pyx
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/hilbert_cython.pyx
  DEPENDS hilbert.pxd
)

add_library(hilbert_cython SHARED 
  hilbert_cython.cpp
)

set_target_properties(hilbert_cython
  PROPERTIES
  PREFIX ""
  OUTPUT_NAME "hilbert_cython"
)

target_include_directories(hilbert_cython
  PRIVATE
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${NUMPY_INCLUDE_DIRS})

target_link_libraries(hilbert_cython
  hilbert)

if(CMAKE_HOST_UNIX)
EXECUTE_PROCESS(COMMAND ln -sf "${CMAKE_CURRENT_BINARY_DIR}/hilbert_cython.so" "${CMAKE_SOURCE_DIR}/hilbert_cython.so")
endif(CMAKE_HOST_UNIX)

add_custom_command(
OUTPUT sur_array_cython.cpp
  COMMAND ${CYTHON} -o sur_array_cython.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sur_array_cython.pyx
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/sur_array_cython.pyx
  DEPENDS sur_array.pxd
)

add_library(sur_array_cython SHARED 
  sur_array_cython.cpp
)

set_target_properties(sur_array_cython
  PROPERTIES
  PREFIX ""
  OUTPUT_NAME "sur_array_cython"
)

target_include_directories(sur_array_cython
  PRIVATE
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${NUMPY_INCLUDE_DIRS})

target_link_libraries(sur_array_cython
  sur_array)

if(CMAKE_HOST_UNIX)
EXECUTE_PROCESS(COMMAND ln -sf "${CMAKE_CURRENT_BINARY_DIR}/sur_array_cython.so" "${CMAKE_SOURCE_DIR}/sur_array_cython.so")
endif(CMAKE_HOST_UNIX)

