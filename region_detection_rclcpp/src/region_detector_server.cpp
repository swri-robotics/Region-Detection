/*
 * @author Jorge Nicho
 * @file region_detector_server.cpp
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

#include <region_detection_msgs/srv/detect_regions.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <cv_bridge/cv_bridge.h>

#include <tf2_eigen/tf2_eigen.h>

#include <pcl/io/pcd_io.h>

#include <pcl_conversions/pcl_conversions.h>

#include <region_detection_core/region_detector.h>

static const std::string REGION_MARKERS_TOPIC = "detected_regions";
static const std::string DETECT_REGIONS_SERVICE = "detect_regions";
static const std::string CLOSED_REGIONS_NS = "closed_regions";

typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > EigenPose3dVector;

static geometry_msgs::msg::Pose pose3DtoPoseMsg(const std::array<float, 6>& p)
{
  using namespace Eigen;
  geometry_msgs::msg::Pose pose_msg;
  Eigen::Affine3d eigen_pose = Translation3d(Vector3d(p[0], p[1], p[2])) * AngleAxisd(p[3], Vector3d::UnitX()) *
                               AngleAxisd(p[4], Vector3d::UnitY()) * AngleAxisd(p[5], Vector3d::UnitZ());

  pose_msg = tf2::toMsg(eigen_pose);
  return std::move(pose_msg);
}

visualization_msgs::msg::MarkerArray convertToAxisMarkers(const std::vector<EigenPose3dVector>& path,
                                                          const std::string& frame_id,
                                                          const std::string& ns,
                                                          const std::size_t& start_id,
                                                          const double& axis_scale,
                                                          const double& axis_length,
                                                          const std::array<float, 6>& offset)
{
  using namespace Eigen;

  visualization_msgs::msg::MarkerArray markers;

  auto create_line_marker = [&](const int id,
                                const std::tuple<float, float, float, float>& rgba) -> visualization_msgs::msg::Marker {
    visualization_msgs::msg::Marker line_marker;
    line_marker.action = line_marker.ADD;
    std::tie(line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a) = rgba;
    line_marker.header.frame_id = frame_id;
    line_marker.type = line_marker.LINE_LIST;
    line_marker.id = id;
    line_marker.lifetime = rclcpp::Duration(0);
    line_marker.ns = ns;
    std::tie(line_marker.scale.x, line_marker.scale.y, line_marker.scale.z) = std::make_tuple(axis_scale, 0.0, 0.0);
    line_marker.pose = pose3DtoPoseMsg(offset);
    return std::move(line_marker);
  };

  // markers for each axis line
  int marker_id = start_id;
  visualization_msgs::msg::Marker x_axis_marker = create_line_marker(++marker_id, std::make_tuple(1.0, 0.0, 0.0, 1.0));
  visualization_msgs::msg::Marker y_axis_marker = create_line_marker(++marker_id, std::make_tuple(0.0, 1.0, 0.0, 1.0));
  visualization_msgs::msg::Marker z_axis_marker = create_line_marker(++marker_id, std::make_tuple(0.0, 0.0, 1.0, 1.0));

  auto add_axis_line = [](const Isometry3d& eigen_pose,
                          const Vector3d& dir,
                          const geometry_msgs::msg::Point& p1,
                          visualization_msgs::msg::Marker& marker) {
    geometry_msgs::msg::Point p2;
    Eigen::Vector3d line_endpoint;

    // axis endpoint
    line_endpoint = eigen_pose * dir;
    std::tie(p2.x, p2.y, p2.z) = std::make_tuple(line_endpoint.x(), line_endpoint.y(), line_endpoint.z());

    // adding line
    marker.points.push_back(p1);
    marker.points.push_back(p2);
  };

  for (auto& poses : path)
  {
    for (auto& pose : poses)
    {
      geometry_msgs::msg::Point p1;
      std::tie(p1.x, p1.y, p1.z) =
          std::make_tuple(pose.translation().x(), pose.translation().y(), pose.translation().z());
      add_axis_line(pose, Vector3d::UnitX() * axis_length, p1, x_axis_marker);
      add_axis_line(pose, Vector3d::UnitY() * axis_length, p1, y_axis_marker);
      add_axis_line(pose, Vector3d::UnitZ() * axis_length, p1, z_axis_marker);
    }
  }

  markers.markers.push_back(x_axis_marker);
  markers.markers.push_back(y_axis_marker);
  markers.markers.push_back(z_axis_marker);
  return std::move(markers);
}

visualization_msgs::msg::MarkerArray
convertToDottedLineMarker(const std::vector<EigenPose3dVector>& path,
                          const std::string& frame_id,
                          const std::string& ns,
                          const std::size_t& start_id = 0,
                          const std::array<double, 4>& rgba_line = { 1.0, 1.0, 0.2, 1.0 },
                          const std::array<double, 4>& rgba_point = { 0.1, .8, 0.2, 1.0 },
                          const std::array<float, 6>& offset = { 0, 0, 0, 0, 0, 0 },
                          const float& line_width = 0.001,
                          const float& point_size = 0.0015)
{
  visualization_msgs::msg::MarkerArray markers_msgs;
  visualization_msgs::msg::Marker line_marker, points_marker;

  // configure line marker
  line_marker.action = line_marker.ADD;
  std::tie(line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a) =
      std::make_tuple(rgba_line[0], rgba_line[1], rgba_line[2], rgba_line[3]);
  line_marker.header.frame_id = frame_id;
  line_marker.type = line_marker.LINE_STRIP;
  line_marker.id = start_id;
  line_marker.lifetime = rclcpp::Duration(0);
  line_marker.ns = ns;
  std::tie(line_marker.scale.x, line_marker.scale.y, line_marker.scale.z) = std::make_tuple(line_width, 0.0, 0.0);
  line_marker.pose = pose3DtoPoseMsg(offset);

  // configure point marker
  points_marker = line_marker;
  points_marker.type = points_marker.POINTS;
  points_marker.ns = ns;
  std::tie(points_marker.color.r, points_marker.color.g, points_marker.color.b, points_marker.color.a) =
      std::make_tuple(rgba_point[0], rgba_point[1], rgba_point[2], rgba_point[3]);
  std::tie(points_marker.scale.x, points_marker.scale.y, points_marker.scale.z) =
      std::make_tuple(point_size, point_size, point_size);

  int id_counter = start_id;
  for (auto& poses : path)
  {
    line_marker.points.clear();
    points_marker.points.clear();
    line_marker.points.reserve(poses.size());
    points_marker.points.reserve(poses.size());
    for (auto& pose : poses)
    {
      geometry_msgs::msg::Point p;
      std::tie(p.x, p.y, p.z) = std::make_tuple(pose.translation().x(), pose.translation().y(), pose.translation().z());
      line_marker.points.push_back(p);
      points_marker.points.push_back(p);
    }

    line_marker.id = (++id_counter);
    points_marker.id = (++id_counter);
    markers_msgs.markers.push_back(line_marker);
    markers_msgs.markers.push_back(points_marker);
  }

  return markers_msgs;
}

class RegionDetectorServer
{
public:
  RegionDetectorServer(std::shared_ptr<rclcpp::Node> node)
    : node_(node), logger_(node->get_logger()), marker_pub_timer_(nullptr)
  {
    // load parameters

    // creating service
    detect_regions_server_ = node->create_service<region_detection_msgs::srv::DetectRegions>(
        DETECT_REGIONS_SERVICE,
        std::bind(&RegionDetectorServer::detectRegionsCallback,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3));

    region_markers_pub_ =
        node->create_publisher<visualization_msgs::msg::MarkerArray>(REGION_MARKERS_TOPIC, rclcpp::QoS(1));

    // run this for verification of parameters
    loadRegionDetectionConfig();
  }

  ~RegionDetectorServer() {}

private:
  region_detection_core::RegionDetectionConfig loadRegionDetectionConfig()
  {
    std::string yaml_config_file = node_->get_parameter("region_detection_cfg_file").as_string();
    return region_detection_core::RegionDetectionConfig::loadFromFile(yaml_config_file);
  }

  void publishRegions(const std::string& frame_id, const std::string ns, const std::vector<EigenPose3dVector>& regions)
  {
    using namespace std::chrono_literals;

    if (marker_pub_timer_)
    {
      marker_pub_timer_->cancel();
    }

    // create markers to publish
    visualization_msgs::msg::MarkerArray region_markers;
    int id = 0;
    for (auto& poses : regions)
    {
      id++;
      visualization_msgs::msg::MarkerArray m =
          convertToAxisMarkers({ poses }, frame_id, ns + std::to_string(id), 0, 0.001, 0.01, { 0, 0, 0, 0, 0, 0 });
      region_markers.markers.insert(region_markers.markers.end(), m.markers.begin(), m.markers.end());

      m = convertToDottedLineMarker({ poses }, frame_id, ns + std::to_string(id));
      region_markers.markers.insert(region_markers.markers.end(), m.markers.begin(), m.markers.end());
    }
    marker_pub_timer_ = node_->create_wall_timer(
        500ms, [this, region_markers]() -> void { region_markers_pub_->publish(region_markers); });
  }

  void detectRegionsCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                             const std::shared_ptr<region_detection_msgs::srv::DetectRegions::Request> request,
                             const std::shared_ptr<region_detection_msgs::srv::DetectRegions::Response> response)
  {
    using namespace region_detection_core;
    using namespace pcl;

    (void)request_header;

    // converting to input for region detection
    RegionDetector::DataBundleVec data_vec;
    const std::string img_name_prefix = "img_input_";
    const std::string pcd_file_prefix = "cloud_input_";
    for (std::size_t i = 0; i < request->clouds.size(); i++)
    {
      RegionDetector::DataBundle data;
      pcl_conversions::toPCL(request->clouds[i], data.cloud_blob);
      cv_bridge::CvImagePtr img = cv_bridge::toCvCopy(request->images[i], sensor_msgs::image_encodings::RGBA8);
      data.image = img->image;
      cv::imwrite(img_name_prefix + std::to_string(i) + ".jpg", data.image);
      pcl::io::savePCDFile(pcd_file_prefix + std::to_string(i) + ".pcd", data.cloud_blob);
      data.transform = tf2::transformToEigen(request->transforms[i]);
      data_vec.push_back(data);
    }

    // region detection
    RegionDetectionConfig config = loadRegionDetectionConfig();
    RegionDetector region_detector(config);
    RegionDetector::RegionResults region_detection_results;
    if (!region_detector.compute(data_vec, region_detection_results))
    {
      response->succeeded = false;
      response->err_msg = "Failed to find closed regions";
      RCLCPP_ERROR_STREAM(logger_, response->err_msg);
      return;
    }
    RCLCPP_INFO(logger_, "Found %i closed regions", region_detection_results.closed_regions_poses.size());

    publishRegions(
        request->transforms.front().header.frame_id, CLOSED_REGIONS_NS, region_detection_results.closed_regions_poses);

    for (const EigenPose3dVector& region : region_detection_results.closed_regions_poses)
    {
      geometry_msgs::msg::PoseArray region_poses;
      for (const Eigen::Affine3d& pose : region)
      {
        region_poses.poses.push_back(tf2::toMsg(pose));
      }
      response->detected_regions.push_back(region_poses);
    }
    response->succeeded = !response->detected_regions.empty();
  }

  // ros interfaces
  rclcpp::Service<region_detection_msgs::srv::DetectRegions>::SharedPtr detect_regions_server_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr region_markers_pub_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Logger logger_;
  rclcpp::TimerBase::SharedPtr marker_pub_timer_;
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
  std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("region_detector", options);
  RegionDetectorServer region_detector(node);
  rclcpp::spin(node);
  return 0;
}
