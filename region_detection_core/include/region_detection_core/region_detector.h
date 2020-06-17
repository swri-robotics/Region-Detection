/*
 * region_detector.h
 *
 *  Created on: Jun 4, 2020
 *      Author: jnicho
 */

#ifndef INCLUDE_REGION_DETECTOR_H_
#define INCLUDE_REGION_DETECTOR_H_

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "region_detection_core/config_types.h"

namespace region_detection_core
{

struct RegionDetectionConfig
{
  // OpenCV configurations
  struct OpenCVCfg
  {
    bool invert_image = true;
    config_2d::ThresholdCfg threshold;
    config_2d::DilationCfg dilation;
    config_2d::CannyCfg canny;
    config_2d::CountourCfg contour;

    bool debug_mode_enable = false;
    std::string debug_window_name = "DEBUG_REGION_DETECTION";
    bool debug_wait_key = false;
  } opencv_cfg;

};

class RegionDetector
{
public:
  struct DataBundle
  {
    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    Eigen::Isometry3d transform;
  };

  typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > EigenPose3dVector;
  struct RegionResults
  {
    std::vector<EigenPose3dVector> region_poses;
  };

  RegionDetector(const RegionDetectionConfig& config, log4cxx::LoggerPtr logger = nullptr);
  RegionDetector(log4cxx::LoggerPtr logger = nullptr);
  virtual ~RegionDetector();

  log4cxx::LoggerPtr getLogger();
  bool configure(const RegionDetectionConfig& config);
  bool configure(const std::string& yaml_str);
  const RegionDetectionConfig& getConfig();

  RegionResults compute(const std::vector<DataBundle>& input);

  static log4cxx::LoggerPtr createDefaultInfoLogger(const std::string& logger_name);
  static log4cxx::LoggerPtr createDefaultDebugLogger(const std::string& logger_name);

private:

  void updateDebugWindow(const cv::Mat& im) const;
  std::tuple<bool,std::string> run2DAnalysis(const RegionDetectionConfig::OpenCVCfg& config, cv::Mat input, std::vector<std::vector<cv::Point> > contours) const;

  log4cxx::LoggerPtr logger_;
  std::shared_ptr<RegionDetectionConfig> cfg_;

};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTOR_H_ */
