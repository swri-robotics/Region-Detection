# Region Detection RosCpp Demo
### Summary
This package contains a demo application meant to demonstrate how to use the capabilities in the main core library

---
### Setup
- Download the [region_detection_test_data.zip](https://github.com/swri-robotics/Region-Detection/wiki/region_detection_test_data.zip) file
- Extract the **region_detection_test_data.zip** file
- CD into the **region_detection_test_data** directory
- Create a shortcut to the directory
	```bash
	ln -s  $(pwd) $HOME/region_detection_data
	```
---
### Run Demo
- Run launch file
	```bash
	roslaunch region_detection_roscpp_demo detect_regions_demo.launch
	```

  You should see Rviz with a visualization of the regions detected.  

- Edit the `detect_regions_demo.launch` launch file and change the config file or the dataset used to see different results
