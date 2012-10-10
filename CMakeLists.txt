cmake_minimum_required(VERSION 2.8)

PROJECT(nn)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "-O2 -g -Wextra -pedantic")

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(GTest)
FIND_PACKAGE(Boost COMPONENTS system random REQUIRED)
#FIND_PACKAGE(LBFGS REQUIRED)

#SET(SPARSENN_SRC src/basic.c src/sparse_nn.c)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(NN_LIB_SRC src/nn.cpp src/optimization.cpp)

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
add_library(nn SHARED ${NN_LIB_SRC})

# ----- Testing -----
IF (GTEST_FOUND)
  ENABLE_TESTING()
  include_directories(${GTEST_INCLUDE_DIR})
  
  add_executable(test_nn test/test_nn.cpp)
  target_link_libraries(test_nn ${GTEST_LIBRARIES} nn ${Boost_RANDOM_LIBRARY} pthread)
 # add_executable(test_rvm tests/rvm_test.cpp)
 # target_link_libraries(test_rvm ${GTEST_LIBRARIES})
  
 # add_executable(rvm_sinc tests/rvm_sinc.cpp)
ENDIF (GTEST_FOUND)
