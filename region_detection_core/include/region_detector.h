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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace region_detection_core
{

struct RegionDetectionConfig
{
  // OpenCV configurations

  struct ThresholdCfg
  {
    bool enable = false;
    int value = 150;
    int type = cv::ThresholdTypes::THRESH_TRUNC;

    static const int MAX_VALUE = 255;
    static const int MAX_TYPE = cv::ThresholdTypes::THRESH_TOZERO_INV;
    static const int MAX_BINARY_VALUE = 255;
  };

  struct DilationCfg
  {
    bool enable = true;
    int elem = 0;
    int size = 1;

    static const int MAX_ELEM = 2;
    static const int MAX_KERNEL_SIZE = 21;
  };

  struct CannyCfg
  {
    bool enable = 1;
    int lower_threshold = 45;
    int upper_threshold = lower_threshold * 3;
    int aperture_size = 1;

    static const int MAX_ENABLE = 1;
    static const int MAX_LOWER_TH = 100;
    static const int MAX_UPPER_TH = 255;
    static const int MAX_APERTURE_SIZE = 3;
  };

  struct CountourCfg
  {
    bool enable = true;
    int mode = CV_RETR_EXTERNAL;
    int method = CV_CHAIN_APPROX_SIMPLE;

    static const int MAX_MODE = CV_RETR_TREE;
    static const int MAX_METHOD = CV_CHAIN_APPROX_TC89_KCOS;
  };

  struct OpenCVCfg
  {
    bool invert_image = true;
    bool enalble_debug_mode = false;
    ThresholdCfg threshold;
    DilationCfg dilation;
    CannyCfg canny;
    CountourCfg contour;
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

  RegionDetector(const RegionDetectionConfig& config, log4cxx::LoggerPtr logger = nullptr);
  RegionDetector(log4cxx::LoggerPtr logger = nullptr);
  virtual ~RegionDetector();

  log4cxx::LoggerPtr getLogger();
  bool configure(const RegionDetectionConfig& config);
  bool configure(const std::string& yaml_str);
  const RegionDetectionConfig& getConfig();

  std::vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d  > > compute(const std::vector<DataBundle>& input);

private:
  log4cxx::LoggerPtr logger_;
  std::shared_ptr<RegionDetectionConfig> cfg_;

};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTOR_H_ */
