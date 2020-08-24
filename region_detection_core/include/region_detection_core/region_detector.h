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
#include <pcl/PCLPointCloud2.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "region_detection_core/config_types.h"

namespace region_detection_core
{
enum class Methods2D : int
{
  GRAYSCALE = 0,
  INVERT,
  THRESHOLD,
  DILATION,
  EROSION,
  CANNY,
  THINNING,
  RANGE,
  HSV,
  EQUALIZE_HIST,
  EQUALIZE_HIST_YUV,
  CLAHE,
};

static const std::map<std::string, Methods2D> METHOD_CODES_MAPPINGS = { { "GRAYSCALE", Methods2D::GRAYSCALE },
                                                                        { "INVERT", Methods2D::INVERT },
                                                                        { "THRESHOLD", Methods2D::THRESHOLD },
                                                                        { "DILATION", Methods2D::DILATION },
                                                                        { "EROSION", Methods2D::EROSION },
                                                                        { "CANNY", Methods2D::CANNY },
                                                                        { "THINNING", Methods2D::THINNING },
                                                                        { "RANGE", Methods2D::RANGE },
                                                                        { "HSV", Methods2D::HSV },
                                                                        { "EQUALIZE_HIST", Methods2D::EQUALIZE_HIST },
                                                                        { "EQUALIZE_HIST_YUV",
                                                                          Methods2D::EQUALIZE_HIST_YUV },
                                                                        { "CLAHE", Methods2D::CLAHE } };

struct RegionDetectionConfig
{
  // OpenCV configurations
  struct OpenCVCfg
  {
    std::vector<std::string> methods = {};

    config_2d::ThresholdCfg threshold;
    config_2d::MorphologicalCfg dilation;
    config_2d::MorphologicalCfg erosion;
    config_2d::CannyCfg canny;
    config_2d::CountourCfg contour;
    config_2d::RangeCfg range;
    config_2d::HSVCfg hsv;
    config_2d::CLAHECfg clahe;

    bool debug_mode_enable = false;
    std::string debug_window_name = "DEBUG_REGION_DETECTION";
    bool debug_wait_key = false;
  } opencv_cfg;

  struct PCL2DCfg
  {
    double downsampling_radius = 4.0;    // pixel units
    double split_dist = 6.0;             // pixel units
    double closed_curve_max_dist = 6.0;  // pixel units
    int simplification_min_points = 10;  // applies simplification only if the closed curve has 10 points or more
    double simplification_alpha = 24.0;  // pixel units, used in concave hull step
  } pcl_2d_cfg;

  struct PCLCfg
  {
    config_3d::StatisticalRemovalCfg stat_removal;
    config_3d::NormalEstimationCfg normal_est;

    double max_merge_dist = 0.01;          /** @brief in meters */
    double closed_curve_max_dist = 0.01;   /** @brief in meters */
    double simplification_min_dist = 0.02; /** @brief in meters */
    double split_dist = 0.1; /** @brief will split segments when the distance between consecutive points exceeds this
                                value, in meters */
    int min_num_points = 10; /** @brief segments must have at least this many points*/

    bool debug_mode_enable = false; /** @brief not used at the moment */
  } pcl_cfg;

  static RegionDetectionConfig loadFromFile(const std::string& yaml_file);
  static RegionDetectionConfig load(const std::string& yaml_str);
};

class RegionDetector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct DataBundle
  {
    cv::Mat image;
    pcl::PCLPointCloud2 cloud_blob;
    Eigen::Isometry3d transform;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  typedef std::vector<DataBundle, Eigen::aligned_allocator<DataBundle>> DataBundleVec;

  typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> EigenPose3dVector;
  struct RegionResults
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
  bool configureFromFile(const std::string& yaml_file);
  const RegionDetectionConfig& getConfig();

  /**
   * @brief computes contours from images
   * @param input             Input image
   * @param contours_indices  Output contours in pixel coordinates
   * @param output            Output image
   * @return  True on success, false otherwise
   */
  bool compute2d(cv::Mat input, cv::Mat& output) const;

  /**
   * @brief computes contours from images and returns their indices
   * @param input             Input image
   * @param contours_indices  Output contours in pixel coordinates
   * @param output            Output image
   * @param contours_indices  The indices of the contours
   * @return  True on success, false otherwise
   */
  bool compute2d(cv::Mat input, cv::Mat& output, std::vector<std::vector<cv::Point>>& contours_indices) const;

  /**
   * @brief computes contours
   * @param input   A vector of data structures containing point clouds and images
   * @param regions (Output) the detected regions
   * @return True on success, false otherwise
   */
  bool compute(const DataBundleVec& input, RegionDetector::RegionResults& regions);

  static log4cxx::LoggerPtr createDefaultInfoLogger(const std::string& logger_name);
  static log4cxx::LoggerPtr createDefaultDebugLogger(const std::string& logger_name);

private:
  /**
   * @class region_detection_core::RegionDetector::Result
   * @brief Convenience class that can be evaluated as a bool and contains an error message, used internally
   */
  struct Result
  {
    /**
     * @brief Constructor for the response object
     * @param success   Set to true if the requested action was completed, use false otherwise.
     * @param data          Optional data that was generated from the requested transaction.
     */
    Result(bool success = true, std::string msg = "") : success(success), msg(msg) {}

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

  // 2d methods
  void updateDebugWindow(const cv::Mat& im) const;

  RegionDetector::Result apply2dCanny(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dDilation(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dErosion(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dThreshold(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dInvert(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dGrayscale(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dRange(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dHSV(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dEqualizeHistYUV(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dEqualizeHist(cv::Mat input, cv::Mat& output) const;
  RegionDetector::Result apply2dCLAHE(cv::Mat input, cv::Mat& output) const;

  Result compute2dContours(cv::Mat input, std::vector<std::vector<cv::Point>>& contours_indices, cv::Mat& output) const;

  // 3d methods

  Result extractContoursFromCloud(const std::vector<std::vector<cv::Point>>& contours_indices,
                                  pcl::PointCloud<pcl::PointXYZ>::ConstPtr input,
                                  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points);

  Result combineIntoClosedRegions(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points,
                                  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves,
                                  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& open_curves);

  Result computePoses(pcl::PointCloud<pcl::PointNormal>::ConstPtr source_normals_cloud,
                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves,
                      std::vector<EigenPose3dVector>& regions);

  Result computeNormals(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_cloud,
                        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& curves_points,
                        std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>& curves_normals);

  Result mergeCurves(pcl::PointCloud<pcl::PointXYZ> c1,
                     pcl::PointCloud<pcl::PointXYZ> c2,
                     pcl::PointCloud<pcl::PointXYZ>& merged);

  pcl::PointCloud<pcl::PointXYZ> sequence(pcl::PointCloud<pcl::PointXYZ>::ConstPtr points, double epsilon = 1e-5);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> split(const pcl::PointCloud<pcl::PointXYZ>& sequenced_points,
                                                         double split_dist);

  void findClosedCurves(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& sequenced_points,
                        double max_dist,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves_vec,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& open_curves_vec);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
  simplifyByMinimunLength(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& segments, double min_length);

  log4cxx::LoggerPtr logger_;
  std::shared_ptr<RegionDetectionConfig> cfg_;
  std::size_t window_counter_;
};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTOR_H_ */
