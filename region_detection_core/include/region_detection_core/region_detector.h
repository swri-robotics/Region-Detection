/*
 * @author Jorge Nicho
 * @file region_detector.h
 * @date Jun 4, 2020
 * @copyright Copyright (c) 2020, Southwest Research Institute
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2020, Southwest Research Institute
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *       * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *       * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *       * Neither the name of the Southwest Research Institute, nor the names
 *       of its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
    // opencv
    bool invert_image = true;
    config_2d::ThresholdCfg threshold;
    config_2d::DilationCfg dilation;
    config_2d::CannyCfg canny;
    config_2d::CountourCfg contour;

    bool debug_mode_enable = false;
    std::string debug_window_name = "DEBUG_REGION_DETECTION";
    bool debug_wait_key = false;
  } opencv_cfg;

  struct PCL2DCfg
  {
    double downsampling_radius = 4.0; // pixel units
    double split_dist = 6.0; // pixel units
    double closed_curve_max_dist = 6.0; // pixel units
    int simplification_min_points = 10; // applies simplification only if the closed curve has 10 points or more
    double simplification_alpha = 24.0; // pixel units, used in concave hull step
  } pcl_2d_cfg;

  struct PCLCfg
  {
    config_3d::StatisticalRemovalCfg stat_removal;
    config_3d::NormalEstimationCfg normal_est;

    double max_merge_dist = 0.01;           /** @brief in meters */
    double closed_curve_max_dist = 0.01;    /** @brief in meters */
    double simplification_min_dist = 0.02;  /** @brief in meters */
    int min_num_points = 10;                /** @brief segments must have at least this many points*/

    bool debug_mode_enable = false;         /** @brief not used at the moment */
  }  pcl_cfg;

};

class RegionDetector
{
public:
  struct DataBundle
  {
    cv::Mat image;
    pcl::PCLPointCloud2 cloud_blob;
    Eigen::Isometry3d transform;
  };

  typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > EigenPose3dVector;
  struct RegionResults
  {
    std::vector<EigenPose3dVector> closed_regions_poses;
    std::vector<EigenPose3dVector> open_regions_poses;

    // additional results
    std::vector<cv::Mat> images;
  };

  RegionDetector(const RegionDetectionConfig& config, log4cxx::LoggerPtr logger = nullptr);
  RegionDetector(log4cxx::LoggerPtr logger = nullptr);
  virtual ~RegionDetector();

  log4cxx::LoggerPtr getLogger();
  bool configure(const RegionDetectionConfig& config);
  bool configure(const std::string& yaml_str);
  const RegionDetectionConfig& getConfig();

  bool compute(const std::vector<DataBundle>& input,RegionDetector::RegionResults& regions);

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

  // 2d methods
  Result compute2dContours(cv::Mat input, std::vector<std::vector<cv::Point> >& contours_indices, cv::Mat& output) const;

  // 3d methods

  Result extractContoursFromCloud(const std::vector<std::vector<cv::Point> >& contours_indices,
                                         pcl::PointCloud<pcl::PointXYZ>::ConstPtr input,
                                         std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points);

  Result combineIntoClosedRegions(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points,
                              std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves,
                              std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& open_curves);

  Result computePoses(pcl::PointCloud<pcl::PointNormal>::ConstPtr source_normals_cloud,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves,
                        std::vector<EigenPose3dVector> & regions);

  Result computeNormals(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_cloud,
                                        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &curves_points,
                                        std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> &curves_normals);

  Result mergeCurves(pcl::PointCloud<pcl::PointXYZ> c1, pcl::PointCloud<pcl::PointXYZ> c2,
                                   pcl::PointCloud<pcl::PointXYZ>& merged);


  pcl::PointCloud<pcl::PointXYZ> sequence(pcl::PointCloud<pcl::PointXYZ>::ConstPtr points,
                                                                  double epsilon = 1e-5);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> split(const pcl::PointCloud<pcl::PointXYZ>& sequenced_points,
                                                                         double split_dist);

  void findClosedCurves(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& sequenced_points, double max_dist,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves_vec,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& open_curves_vec);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> simplifyByMinimunLength(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&
                                                                                      segments, double min_length);



  log4cxx::LoggerPtr logger_;
  std::shared_ptr<RegionDetectionConfig> cfg_;
  std::size_t window_counter_;

};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTOR_H_ */
