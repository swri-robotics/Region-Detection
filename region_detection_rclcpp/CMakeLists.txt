cmake_minimum_required(VERSION 3.5)
project(region_detection_rclcpp)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(PCL REQUIRED COMPONENTS io)

# find dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(region_detection_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(region_detection_core REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_eigen REQUIRED)

### Build
add_executable(interactive_region_selection src/interactive_region_selection.cpp)
target_include_directories(interactive_region_selection PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(interactive_region_selection SYSTEM PUBLIC)
target_link_libraries(interactive_region_selection 
  region_detection_core::region_detection_core)
ament_target_dependencies(interactive_region_selection
  rclcpp rclcpp_components
  region_detection_msgs
  visualization_msgs
  tf2_eigen
  std_msgs
  interactive_markers)
  
add_executable(region_detector_server src/region_detector_server.cpp)
target_include_directories(region_detector_server PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(region_detector_server SYSTEM PUBLIC)
target_link_libraries(region_detector_server 
  ${PCL_LIBRARIES}
  region_detection_core::region_detection_core)
ament_target_dependencies(region_detector_server
  rclcpp rclcpp_components
  region_detection_msgs
  visualization_msgs
  pcl_conversions
  cv_bridge
  tf2_eigen
  std_msgs
  sensor_msgs)
  
add_executable(crop_data_server src/crop_data_server.cpp)
target_include_directories(crop_data_server PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(crop_data_server SYSTEM PUBLIC)
target_link_libraries(crop_data_server 
  region_detection_core::region_detection_core)
ament_target_dependencies(crop_data_server
  rclcpp rclcpp_components
  region_detection_msgs
  pcl_conversions
  tf2_eigen
  std_msgs
  sensor_msgs)
  
  
### Install
install(TARGETS interactive_region_selection region_detector_server crop_data_server
DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
