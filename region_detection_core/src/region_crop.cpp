/*
 * @author Jorge Nicho
 * @file region_crop.cpp
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
#include <numeric>

#include "region_detection_core/region_crop.h"

#include <pcl/impl/instantiate.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

#include <console_bridge/console.h>

#include <boost/format.hpp>
#include <boost/make_shared.hpp>

static const double EPSILON = 1e-8;

typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> EigenPose3dVector_HIDDEN;

pcl::PointCloud<pcl::PointXYZ>::Ptr computePlanarHullFromNormals(EigenPose3dVector_HIDDEN region_3d)
{
  using namespace pcl;
  using namespace Eigen;

  PointCloud<PointXYZ>::Ptr planar_hull = boost::make_shared<PointCloud<PointXYZ>>();

  // compute plane normal from averages
  pcl::Normal plane_normal;
  Vector3d normal_vec = std::accumulate(region_3d.begin(),
                                        region_3d.end(),
                                        Vector3d(0, 0, 0),
                                        [](Vector3d current_normal, Eigen::Isometry3d pose) -> Vector3d {
                                          Vector3d pose_normal;
                                          pose_normal.array() = pose.linear().col(2);
                                          pose_normal += current_normal;
                                          return pose_normal;
                                        });
  normal_vec /= region_3d.size();
  normal_vec.normalize();

  // compute centroid
  Vector3d centroid = std::accumulate(region_3d.begin(),
                                      region_3d.end(),
                                      Vector3d(0, 0, 0),
                                      [](Vector3d current_centroid, const Eigen::Isometry3d& pose) {
                                        Vector3d region_location;
                                        region_location = pose.translation();
                                        region_location += current_centroid;
                                        return region_location;
                                      });
  centroid /= region_3d.size();

  // defining plane
  double d = -centroid.dot(normal_vec);
  pcl::ModelCoefficients::Ptr coefficients = boost::make_shared<pcl::ModelCoefficients>();
  coefficients->values.resize(4);
  coefficients->values[0] = normal_vec(0);
  coefficients->values[1] = normal_vec(1);
  coefficients->values[2] = normal_vec(2);
  coefficients->values[3] = d;

  // converting to point cloud
  PointCloud<PointXYZ>::Ptr region_cloud_3d = boost::make_shared<PointCloud<PointXYZ>>();
  std::transform(
      region_3d.begin(), region_3d.end(), std::back_inserter(*region_cloud_3d), [](const Eigen::Isometry3d& pose) {
        PointXYZ p;
        p.getArray3fMap() = pose.translation().array().cast<float>();
        return p;
      });

  // projecting onto plane
  pcl::ProjectInliers<PointXYZ> project_inliers;
  project_inliers.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  project_inliers.setInputCloud(region_cloud_3d);
  project_inliers.setModelCoefficients(coefficients);
  project_inliers.filter(*planar_hull);

  return planar_hull;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr computePlanarHullFromPlane(const EigenPose3dVector_HIDDEN& region_3d)
{
  using namespace pcl;
  using namespace Eigen;

  PointCloud<PointXYZ>::Ptr planar_hull = boost::make_shared<PointCloud<PointXYZ>>();

  // converting to point cloud
  PointCloud<PointXYZ>::Ptr region_cloud_3d = boost::make_shared<PointCloud<PointXYZ>>();
  std::transform(
      region_3d.begin(), region_3d.end(), std::back_inserter(*region_cloud_3d), [](const Eigen::Isometry3d& pose) {
        PointXYZ p;
        p.getArray3fMap() = pose.translation().array().cast<float>();
        return p;
      });

  // computing moi
  PointXYZ min_point, max_point, center_point;
  Matrix3f rotation;
  Eigen::Vector3f center, x_axis, y_axis, z_axis;
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> moi;
  moi.setInputCloud(region_cloud_3d);
  moi.compute();
  moi.getOBB(min_point, max_point, center_point, rotation);

  // computing distance threshold
  Vector3f diff = max_point.getArray3fMap() - min_point.getArray3fMap();
  double distance_threshold = diff.array().abs().maxCoeff();

  // computing plane
  PointIndices indices;
  pcl::ModelCoefficients::Ptr coefficients = boost::make_shared<pcl::ModelCoefficients>();
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(distance_threshold);
  seg.setInputCloud(region_cloud_3d);
  seg.segment(indices, *coefficients);

  // projecting onto plane
  pcl::ProjectInliers<PointXYZ> project_inliers;
  project_inliers.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  project_inliers.setInputCloud(region_cloud_3d);
  project_inliers.setModelCoefficients(coefficients);
  project_inliers.filter(*planar_hull);

  return planar_hull;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr computePlanarHullFromZVector(const EigenPose3dVector_HIDDEN& region_3d)
{
  using namespace pcl;
  using namespace Eigen;

  PointCloud<PointXYZ>::Ptr planar_hull = boost::make_shared<PointCloud<PointXYZ>>();

  // converting to point cloud
  PointCloud<PointXYZ>::Ptr region_cloud_3d = boost::make_shared<PointCloud<PointXYZ>>();
  std::transform(
      region_3d.begin(), region_3d.end(), std::back_inserter(*region_cloud_3d), [](const Eigen::Isometry3d& pose) {
        PointXYZ p;
        p.getArray3fMap() = pose.translation().array().cast<float>();
        return p;
      });

  // computing moi
  Eigen::Vector3f center, x_axis, y_axis, z_axis;
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> moi;
  moi.setInputCloud(region_cloud_3d);
  moi.compute();
  moi.getEigenVectors(x_axis, y_axis, z_axis);
  moi.getMassCenter(center);

  // defining plane
  Vector3f normal_vec = z_axis.normalized();
  double d = -center.dot(normal_vec);
  pcl::ModelCoefficients::Ptr coefficients = boost::make_shared<pcl::ModelCoefficients>();
  coefficients->values.resize(4);
  coefficients->values[0] = normal_vec(0);
  coefficients->values[1] = normal_vec(1);
  coefficients->values[2] = normal_vec(2);
  coefficients->values[3] = d;

  // projecting onto plane
  pcl::ProjectInliers<PointXYZ> project_inliers;
  project_inliers.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  project_inliers.setInputCloud(region_cloud_3d);
  project_inliers.setModelCoefficients(coefficients);
  project_inliers.filter(*planar_hull);

  return planar_hull;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr computePlanarHullFromUserVector(const EigenPose3dVector_HIDDEN& region_3d,
                                                                    const Eigen::Vector3d& user_vec)
{
  using namespace pcl;
  using namespace Eigen;

  PointCloud<PointXYZ>::Ptr planar_hull = boost::make_shared<PointCloud<PointXYZ>>();

  // converting to point cloud
  PointCloud<PointXYZ>::Ptr region_cloud_3d = boost::make_shared<PointCloud<PointXYZ>>();
  std::transform(
      region_3d.begin(), region_3d.end(), std::back_inserter(*region_cloud_3d), [](const Eigen::Isometry3d& pose) {
        PointXYZ p;
        p.getArray3fMap() = pose.translation().array().cast<float>();
        return p;
      });

  // computing moi
  Eigen::Vector3f center;
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> moi;
  moi.setInputCloud(region_cloud_3d);
  moi.compute();
  moi.getMassCenter(center);

  // defining plane
  Vector3f normal_vec = user_vec.normalized().cast<float>();
  double d = -center.dot(normal_vec);
  pcl::ModelCoefficients::Ptr coefficients = boost::make_shared<pcl::ModelCoefficients>();
  coefficients->values.resize(4);
  coefficients->values[0] = normal_vec(0);
  coefficients->values[1] = normal_vec(1);
  coefficients->values[2] = normal_vec(2);
  coefficients->values[3] = d;

  // projecting onto plane
  pcl::ProjectInliers<PointXYZ> project_inliers;
  project_inliers.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  project_inliers.setInputCloud(region_cloud_3d);
  project_inliers.setModelCoefficients(coefficients);
  project_inliers.filter(*planar_hull);

  return planar_hull;
}

template <typename PointT>
void scaleCloud(const double scale_factor, typename pcl::PointCloud<PointT>& cloud)
{
  Eigen::Vector4f centroid;
  pcl::PointCloud<PointT> demeaned_cloud;
  pcl::compute3DCentroid(cloud, centroid);
  pcl::demeanPointCloud(cloud, centroid, demeaned_cloud);

  // scaling all points now
  for (std::size_t i = 0; i < demeaned_cloud.size(); i++)
  {
    PointT p = demeaned_cloud[i];
    p.x = scale_factor * p.x;
    p.y = scale_factor * p.y;
    p.z = scale_factor * p.z;
    demeaned_cloud[i] = p;
  }

  // transforming demeaned cloud back to original centroid
  Eigen::Affine3f transform = pcl::getTransformation(centroid.x(), centroid.y(), centroid.z(), 0, 0, 0);
  pcl::transformPointCloud(demeaned_cloud, cloud, transform);
}

namespace region_detection_core
{
template <typename PointT>
RegionCrop<PointT>::RegionCrop() : input_(nullptr)
{
}

template <typename PointT>
RegionCrop<PointT>::~RegionCrop()
{
}

template <typename PointT>
inline void RegionCrop<PointT>::setRegion(const EigenPose3dVector_HIDDEN& closed_region)
{
  using namespace Eigen;
  Vector3d p0, pf;
  p0 = closed_region.front().translation();
  pf = closed_region.back().translation();

  double diff = (pf - p0).norm();
  if (diff > EPSILON)
  {
    throw std::runtime_error("region end points are too far from each other, region isn't closed");
  }
  closed_region_ = closed_region;
}

template <typename PointT>
inline void RegionCrop<PointT>::setConfig(const RegionCropConfig& config)
{
  // check method
  static const std::vector<DirectionEstMethods> valid_options = { DirectionEstMethods::NORMAL_AVGR,
                                                                  DirectionEstMethods::PLANE_NORMAL,
                                                                  DirectionEstMethods::POSE_Z_AXIS,
                                                                  DirectionEstMethods::USER_DEFINED };
  DirectionEstMethods method_flag = config.dir_estimation_method;
  if (std::find(valid_options.begin(), valid_options.end(), method_flag) == valid_options.end())
  {
    std::string err_msg = boost::str(boost::format("The flag %i is not a supported Direction Estimation Method") %
                                     static_cast<int>(config.dir_estimation_method));
    throw std::runtime_error(err_msg);
  }

  // check height limits
  if (config.heigth_limits.first >= config.heigth_limits.second)
  {
    throw std::runtime_error("Height Limits min value is greater than or equal to max");
  }

  config_ = config;
}

template <typename PointT>
void RegionCrop<PointT>::setInput(const typename pcl::PointCloud<PointT>::ConstPtr& cloud)
{
  input_ = cloud;
}

template <typename PointT>
std::vector<int> region_detection_core::RegionCrop<PointT>::filter(bool reverse)
{
  using namespace pcl;
  if (!input_)
  {
    throw std::runtime_error("Input cloud pointer is null");
  }

  // creating planar hull
  PointCloud<PointXYZ>::Ptr planar_hull = boost::make_shared<PointCloud<PointXYZ>>();

  switch (config_.dir_estimation_method)
  {
    case DirectionEstMethods::NORMAL_AVGR:
      planar_hull = computePlanarHullFromNormals(closed_region_);
      break;
    case DirectionEstMethods::PLANE_NORMAL:
      planar_hull = computePlanarHullFromPlane(closed_region_);
      break;
    case DirectionEstMethods::POSE_Z_AXIS:
      planar_hull = computePlanarHullFromZVector(closed_region_);
      break;
    case DirectionEstMethods::USER_DEFINED:
      planar_hull = computePlanarHullFromUserVector(closed_region_, config_.user_dir);
      break;
    default:
      std::string err_msg = boost::str(boost::format("Direction Estimation Method %i is not supported") %
                                       static_cast<int>(config_.dir_estimation_method));
      throw std::runtime_error(err_msg);
  }

  // scaling planar hull
  scaleCloud(config_.scale_factor, *planar_hull);

  // extracting region within polygon
  PointIndices inlier_indices;
  typename PointCloud<PointT>::Ptr planar_hull_t = boost::make_shared<PointCloud<PointT>>();
  pcl::copyPointCloud(*planar_hull, *planar_hull_t);
  ExtractPolygonalPrismData<PointT> extract_polygon;
  extract_polygon.setHeightLimits(config_.heigth_limits.first, config_.heigth_limits.second);
  extract_polygon.setViewPoint(config_.view_point.x(), config_.view_point.y(), config_.view_point.z());
  extract_polygon.setInputPlanarHull(planar_hull_t);
  extract_polygon.setInputCloud(input_);
  extract_polygon.segment(inlier_indices);

  std::vector<int> indices_vec = inlier_indices.indices;
  if (reverse)
  {
    std::vector<int> all_indices_vec, diff;
    all_indices_vec.resize(input_->size());
    std::iota(all_indices_vec.begin(), all_indices_vec.end(), 0);
    std::sort(indices_vec.begin(), indices_vec.end());
    std::set_difference(all_indices_vec.begin(),
                        all_indices_vec.end(),
                        indices_vec.begin(),
                        indices_vec.end(),
                        std::inserter(diff, diff.begin()));
    indices_vec = diff;
  }

  return indices_vec;
}

#define PCL_INSTANTIATE_RegionCrop(T) template class PCL_EXPORTS RegionCrop<T>;

PCL_INSTANTIATE(RegionCrop, PCL_XYZ_POINT_TYPES);

} /* namespace region_detection_core */
