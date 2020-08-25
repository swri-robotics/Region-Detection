# Region Detection Rclcpp
### Summary
This package contains ROS2  nodes that make the functionality in the `region_detection_core` package available to a ROS2 environment.

---
### Nodes
#### region_detector_server: 
Detects contours from 2d images and 3d point clouds
- Parameters:
  - region_detection_cfg_file: absolute path the the config file
- Services
  - detect_regions: service that detects the contours of the regions found in the input images and point clouds.
- Publications:
  - detected_regions: Marker arrays that help visualize the detected regions

#### interactive_region_selection
Shows the region contours as clickable interactive markers in Rviz
- Parameters:
  - region_height: the height of each region shown
- Services:
  - show_selectable_regions: Shows the specified regions as clickable markers in rviz
  - get_selected_regions: Returns the indices of the regions that are currently selected

#### crop_data_server
Crops data using the region contours as a boundary
- Parameters:
  - region_crop: absolute path to the configuration file containing the configuration parameters 
- Services:
  - crop_data: crops the data that falls outside the region contour.