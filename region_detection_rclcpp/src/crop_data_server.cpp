/*
 * @author Jorge Nicho
 * @file crop_data_server.cpp
 * @date Aug 4, 2020
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

#include <rclcpp/rclcpp.hpp>

#include <region_detection_msgs/srv/crop_data.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <cv_bridge/cv_bridge.h>

#include <tf2_eigen/tf2_eigen.h>

#include <pcl_conversions/pcl_conversions.h>

#include <region_detection_core/region_crop.h>

static const std::string CROP_DATA_SERVICE = "crop_data";

class CropDataServer
{
public:
  CropDataServer(std::shared_ptr<rclcpp::Node> node) : node_(node), logger_(node->get_logger())
  {
    // creating service
    crop_data_server_ =
        node->create_service<region_detection_msgs::srv::CropData>(CROP_DATA_SERVICE,
                                                                   std::bind(&CropDataServer::cropDataCallback,
                                                                             this,
                                                                             std::placeholders::_1,
                                                                             std::placeholders::_2,
                                                                             std::placeholders::_3));

    // check parameters
    loadRegionCropConfig();
  }

  ~CropDataServer() {}

private:
  region_detection_core::RegionCropConfig loadRegionCropConfig()
  {
    region_detection_core::RegionCropConfig cfg;
    const std::string param_ns = "region_crop.";
    cfg.scale_factor = node_->get_parameter(param_ns + "scale_factor").as_double();
    cfg.plane_dist_threshold = node_->get_parameter(param_ns + "plane_dist_threshold").as_double();
    double heigth_limits_min = node_->get_parameter(param_ns + "heigth_limits_min").as_double();
    double heigth_limits_max = node_->get_parameter(param_ns + "heigth_limits_max").as_double();
    cfg.dir_estimation_method = static_cast<region_detection_core::DirectionEstMethods>(
        node_->get_parameter(param_ns + "dir_estimation_method").as_int());
    std::vector<double> user_dir = node_->get_parameter(param_ns + "user_dir").as_double_array();
    std::vector<double> view_point = node_->get_parameter(param_ns + "view_point").as_double_array();

    cfg.heigth_limits = std::make_pair(heigth_limits_min, heigth_limits_max);
    cfg.user_dir = Eigen::Map<Eigen::Vector3d>(user_dir.data(), user_dir.size());
    cfg.view_point = Eigen::Map<Eigen::Vector3d>(view_point.data(), view_point.size());

    return cfg;
  }

  void cropDataCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                        const std::shared_ptr<region_detection_msgs::srv::CropData::Request> request,
                        const std::shared_ptr<region_detection_msgs::srv::CropData::Response> response)
  {
    using namespace region_detection_core;
    using namespace region_detection_msgs::msg;
    using namespace pcl;
    using Cloud = PointCloud<PointXYZ>;

    (void)request_header;

    // use detected regions to crop
    RegionCropConfig region_crop_cfg = loadRegionCropConfig();
    RegionCrop<pcl::PointXYZ> region_crop;
    region_crop.setConfig(region_crop_cfg);
    for (std::size_t i = 0; i < request->crop_regions.size(); i++)
    {
      // converting region into datatype
      RegionCrop<pcl::PointXYZ>::EigenPose3dVector crop_region;
      const geometry_msgs::msg::PoseArray& crop_region_poses = request->crop_regions[i];
      std::transform(crop_region_poses.poses.begin(),
                     crop_region_poses.poses.end(),
                     std::back_inserter(crop_region),
                     [](const geometry_msgs::msg::Pose& pose) {
                       Eigen::Isometry3d eig_pose;
                       tf2::fromMsg(pose, eig_pose);
                       return eig_pose;
                     });

      PoseSet cropped_dataset;
      region_crop.setRegion(crop_region);
      for (std::size_t j = 0; j < request->input_data.size(); j++)
      {
        const geometry_msgs::msg::PoseArray& segment_poses = request->input_data[j];

        Cloud::Ptr segments_points = boost::make_shared<Cloud>();
        segments_points->reserve(segment_poses.poses.size());

        std::transform(segment_poses.poses.begin(),
                       segment_poses.poses.end(),
                       std::back_inserter(*segments_points),
                       [](const geometry_msgs::msg::Pose& pose) {
                         PointXYZ p;
                         p.x = pose.position.x;
                         p.y = pose.position.y;
                         p.z = pose.position.z;
                         return p;
                       });

        region_crop.setInput(segments_points);
        std::vector<int> inlier_indices = region_crop.filter();

        if (inlier_indices.empty())
        {
          continue;
        }
        RCLCPP_INFO(logger_, "Found %i inliers in segment %j within region %i", inlier_indices.size(), j, i);

        // extracting poinst in raster
        geometry_msgs::msg::PoseArray cropped_segment_poses;
        std::for_each(inlier_indices.begin(), inlier_indices.end(), [&cropped_segment_poses, &segment_poses](int idx) {
          cropped_segment_poses.poses.push_back(segment_poses.poses[idx]);
        });
        cropped_dataset.pose_arrays.push_back(cropped_segment_poses);
      }
      response->cropped_data.push_back(cropped_dataset);
    }

    if (response->cropped_data.empty())
    {
      response->succeeded = false;
      response->err_msg = "Failed to crop toolpaths";
      RCLCPP_ERROR_STREAM(logger_, response->err_msg);
    }

    response->succeeded = true;
  }

  // ros interfaces
  rclcpp::Service<region_detection_msgs::srv::CropData>::SharedPtr crop_data_server_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Logger logger_;
};

int main(int argc, char** argv)
{
  // force flush of the stdout buffer.
  // this ensures a correct sync of all prints
  // even when executed simultaneously within the launch file.
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("crop_data_server", options);
  CropDataServer region_detector(node);
  rclcpp::spin(node);
  return 0;
}
