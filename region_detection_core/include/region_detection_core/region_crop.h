/*
 * @author Jorge Nicho
 * @file region_crop.h
 * @date Jul 10, 2020
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

#ifndef INCLUDE_REGION_DETECTION_CORE_REGION_CROP_H_
#define INCLUDE_REGION_DETECTION_CORE_REGION_CROP_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Geometry>

namespace region_detection_core
{
enum class DirectionEstMethods : unsigned int
{
  PLANE_NORMAL = 1,
  NORMAL_AVGR,
  POSE_Z_AXIS,
  USER_DEFINED
};

struct RegionCropConfig
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double scale_factor = 1.0;
  double plane_dist_threshold = 0.1;
  std::pair<double, double> heigth_limits = std::make_pair(-0.1, 0.1);
  DirectionEstMethods dir_estimation_method = DirectionEstMethods::PLANE_NORMAL;
  Eigen::Vector3d user_dir = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d view_point = Eigen::Vector3d(0, 0, 10.0);
};

template <typename PointT>
class RegionCrop
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > EigenPose3dVector;

  RegionCrop();
  virtual ~RegionCrop();

  void setConfig(const RegionCropConfig& config);
  void setRegion(const EigenPose3dVector& closed_region);
  void setInput(const typename pcl::PointCloud<PointT>::ConstPtr& cloud);
  std::vector<int> filter(bool reverse = false);

private:
  EigenPose3dVector closed_region_;
  RegionCropConfig config_;
  typename pcl::PointCloud<PointT>::ConstPtr input_;
};

} /* namespace region_detection_core */

#endif /* INCLUDE_REGION_DETECTION_CORE_REGION_CROP_H_ */
