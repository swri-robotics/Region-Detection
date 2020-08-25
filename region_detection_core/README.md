
# Region Detection Core
### Summary
This package contains the main library used for detecting the contours of hand-drawn regions made with a marker on a surface.  

---
### RegionDetector:  
This is the main class implementation and takes 2d images and 3d point clouds as inputs and returns the 3d locations and of the points encompassing the detected contours.  The color of the contours shall be dark and in high contrast with the surface.  The images and point clouds are assumed to be of the same size so if the image is 480 x 640 then the point cloud size should match that.

- Configuration
The configuration file needed by the region detection contains various fields to configure the opencv and pcl filters. See [here](config/config.yaml) for an example
  - opencv:  This section contains the **methods** field which is a list of filter methods to apply to the image; the order in which these are applied are from left to right.  Following this are the configuration parameters specific to each method supported.  The method supported as of now are the following:
    - INVERT
    - GRAYSCALE       # converts the image into a single channel grayscale
    - THRESHOLD
    - DILATION
    - EROSION
    - CANNY
    - THINNING
    - RANGE
    - CLAHE
    - EQUALIZE_HIST
    - EQUALIZE_HIST_YUV
    - HSV
  - pcl2d:  These are parameters used to configure various pcl filters.  These filters are applied in pixel space and assume the the **z** value of each point is 0.
  - pcl:  These are parameters used to configure various pcl filters.  These filters are applied to the 3d data in the point cloud that corresponds to the contours detected in the 2d analysis.
---

### RegionCrop:   
This class crops data outside the contour of the regions, the reverse is also possible.
- Configuration
The configuration yaml file shall have the following form
```yaml
scale_factor: 1.2
plane_dist_threshold: 0.2
heigth_limits_min: -0.1
heigth_limits_max: 0.1
dir_estimation_method: 1       # PLANE_NORMAL: 1, NORMAL_AVGR: 2, POSE_Z_AXIS: 3, USER_DEFINED: 4
user_dir: [0.0, 0.0, 0.1]      # Used when dir_estimation_method == 4
view_point: [0.0, 0.0, 10.0]
```

---
### Test Program
The `region_detection_test` program leverages the `RegionDetector` class and takes a configuration and image file in order to detect the contours in the image.  In order to run this program do the following:
- Locate the executable:
  - For a colcon build, it should be in the `install/region_detection_core/bin/` directory
  - For a catkin build, it should be inside the `devel/bin` directory
- Locate a yaml configuration and image file.
- Run the following command
  ```bash
  ./install/region_detection_core/bin/region_detection_test <absolute/path/to/config/file> <absolute/path/to/image/file>
  ```
- In addition to that, a third optional  argument can be passed in order to run the contour detection function after applying the opencv filters.  If `1` is used then the contours will be drawn on the image with different colors to distinguish them apart. 

