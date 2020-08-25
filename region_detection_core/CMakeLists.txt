cmake_minimum_required(VERSION 3.5.0)
project(region_detection_core VERSION 0.0.0 LANGUAGES CXX)

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
find_package(Eigen3 REQUIRED)
find_package(console_bridge REQUIRED)
find_package(yaml-cpp REQUIRED )

find_package(PkgConfig REQUIRED)
pkg_check_modules(yaml_cpp REQUIRED yaml-cpp)


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
 src/region_crop.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Log4cxx_LIBRARY}
  yaml-cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${OPENCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Log4cxx_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
) 

add_executable(region_detection_test 
  src/tests/region_detection_test.cpp)
target_link_libraries(region_detection_test 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${PROJECT_NAME})

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
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)

install(TARGETS threshold_grayscale_test threshold_in_range_test adaptive_threshold_test region_detection_test
	DESTINATION bin)

list (APPEND PACKAGE_LIBRARIES ${PROJECT_NAME})
install(TARGETS ${PACKAGE_LIBRARIES}
  EXPORT ${PROJECT_NAME}-targets
  DESTINATION lib)
  
install(EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}:: DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)
  
# Create cmake config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME})

export(EXPORT ${PROJECT_NAME}-targets NAMESPACE ${PROJECT_NAME}:: FILE
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake)

#############
## Testing ##
#############

