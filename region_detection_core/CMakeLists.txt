cmake_minimum_required(VERSION 3.0.2)
project(region_detection_core)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-w -Wextra -Wpedantic)
endif()


## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(PCL REQUIRED COMPONENTS common filters surface segmentation)
find_package(Eigen REQUIRED)
find_package(yaml-cpp REQUIRED)

## Logging library
if( Log4cxx_DIR )
  find_package( Log4cxx NO_MODULE )
elseif( NOT Log4cxx_FOUND )
  message(STATUS "Searching for log4cxx/logger.h")
  find_path( Log4cxx_INCLUDE_DIR log4cxx/logger.h )

  message(STATUS "Searching for libLog4cxx")
  find_library( Log4cxx_LIBRARY log4cxx )

  include( FindPackageHandleStandardArgs )
  FIND_PACKAGE_HANDLE_STANDARD_ARGS( Log4cxx Log4cxx_INCLUDE_DIR Log4cxx_LIBRARY )
  if( LOG4CXX_FOUND )
    set( Log4cxx_FOUND TRUE )
  endif()
endif()

###########
## Build ##
###########

# check if cxx_std_14 is found in the CMAKE_CXX_COMPILE_FEATURES list, set CXX_FEATURE_FOUND=-1 when it does not
list(FIND CMAKE_CXX_COMPILE_FEATURES cxx_std_14 CXX_FEATURE_FOUND)

## Specify additional locations of header files
include_directories(
  ${OPENCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PCL_LIBRARIES}
  ${Log4cxx_INCLUDE_DIRS}  
)

# build executables
add_executable(threshold_grayscale_test src/tests/threshold_grayscale_test.cpp)
target_link_libraries(threshold_grayscale_test 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})
  
add_executable(threshold_in_range_test src/tests/threshold_in_range_test.cpp)
target_link_libraries(threshold_in_range_test 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})
  
add_executable(adaptive_threshold_test src/tests/adaptive_threshold_test.cpp)
target_link_libraries(adaptive_threshold_test 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})

# build library
add_library(${PROJECT_NAME} SHARED
 src/region_detector.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Log4cxx_LIBRARY}
  ${yaml-cpp_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${OPENCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_LIBRARIES}
  ${Log4cxx_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
) 

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
if(CXX_FEATURE_FOUND EQUAL "-1")
  target_compile_options(${PROJECT_NAME} PUBLIC -std=c++14)
else()
  target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
endif()



#############
## Install ##
#############

install(DIRECTORY include/
  DESTINATION include
)

install(FILES package.xml DESTINATION share/${PROJECT_NAME})


#############
## Testing ##
#############

