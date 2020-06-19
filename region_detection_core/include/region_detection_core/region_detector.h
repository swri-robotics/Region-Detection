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

  struct PCLCfg
  {
    config_3d::DownsampleCfg downsample;
    config_3d::OrderingCfg ordering;

    double max_merge_dist = 0.01;

    bool debug_mode_enable = false;
  }  pcl_cfg;

};

class RegionDetector
{
public:
  struct DataBundle
  {
    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
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

  struct Result
  {
    /**
     * @brief Constructor for the response object
     * @param success   Set to true if the requested action was completed, use false otherwise.
     * @param data          Optional data that was generated from the requested transaction.
     */
    Result(bool success = true, std::string msg = "")
      : success(success), msg(msg)
    {
    }

    Result(const Result& obj) : success(obj.success), msg(obj.msg) {}

    ~Result() {}

    Result& operator=(const Result& obj)
    {
      success = obj.success;
      msg = obj.msg;
      return *this;
    }

    Result& operator=(const bool& b)
    {
      this->success = b;
      return *this;
    }

    operator bool() const { return success; }

    bool success;
    std::string msg;
  };

  void updateDebugWindow(const cv::Mat& im) const;
  Result logAndReturn(bool success, const std::string& err_msg) const;

  Result compute2dContours(const RegionDetectionConfig::OpenCVCfg& config,
                                             cv::Mat input, std::vector<std::vector<cv::Point> > contours_indices) const;

  Result extractContoursFromCloud(const std::vector<std::vector<cv::Point> > contours_indices,
                                         pcl::PointCloud<pcl::PointXYZ>::ConstPtr input,
                                         std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points);


  Result computeClosedRegions(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points,
                                                    std::vector<EigenPose3dVector>& region_poses);

  Result mergeCurves(pcl::PointCloud<pcl::PointXYZ> c1, pcl::PointCloud<pcl::PointXYZ> c2,
                                   pcl::PointCloud<pcl::PointXYZ>& merged);

  Result reorder(pcl::PointCloud<pcl::PointXYZ>::ConstPtr points,
                               pcl::PointCloud<pcl::PointXYZ>& ordered_points);

  log4cxx::LoggerPtr logger_;
  std::shared_ptr<RegionDetectionConfig> cfg_;

};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTOR_H_ */
