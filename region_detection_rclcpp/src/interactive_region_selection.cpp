/*
 * @author Jorge Nicho
 * @file interactive_region_selection.cpp
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

#include <boost/bind.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <cv_bridge/cv_bridge.h>

#include <tf2_eigen/tf2_eigen.h>

#include <interactive_markers/interactive_marker_server.hpp>

#include <geometry_msgs/msg/pose_array.hpp>

#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/menu_entry.hpp>

#include <region_detection_core/region_detector.h>
#include <region_detection_core/region_crop.h>

#include <region_detection_msgs/srv/show_selectable_regions.hpp>
#include <region_detection_msgs/srv/get_selected_regions.hpp>

static const std::string REGION_ID_PREFIX = "region_";
static const std::string SHOW_SELECTABLE_REGIONS_SERVICE = "show_selectable_regions";
static const std::string GET_SELECTED_REGIONS_SERVICE = "get_selected_regions";

static const std::string DEFAULT_FRAME_ID = "world";

namespace selection_colors_rgba
{
using RGBA = std::tuple<double, double, double, double>;
static const RGBA SELECTED = { 1.0, 1.0, 0.0, 1 };
static const RGBA UNSELECTED = { 0.3, 0.3, 0.3, 1 };
}  // namespace selection_colors_rgba

class InteractiveRegionManager
{
public:
  InteractiveRegionManager(std::shared_ptr<rclcpp::Node> node)
    : node_(node), interactive_marker_server_("InteractiveRegionManager", node)
  {
    // loading parameters
    region_height_ = node_->get_parameter("region_height").as_double();

    // creating service
    show_selectable_regions_server_ = node->create_service<region_detection_msgs::srv::ShowSelectableRegions>(
        SHOW_SELECTABLE_REGIONS_SERVICE,
        std::bind(&InteractiveRegionManager::showSelectableRegionsCallback,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3));
    get_selected_regions_server_ = node->create_service<region_detection_msgs::srv::GetSelectedRegions>(
        GET_SELECTED_REGIONS_SERVICE,
        std::bind(&InteractiveRegionManager::getSelectedRegionsCallback,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3));
  }

  ~InteractiveRegionManager() {}

private:
  void showSelectableRegionsCallback(
      const std::shared_ptr<rmw_request_id_t> request_header,
      const std::shared_ptr<region_detection_msgs::srv::ShowSelectableRegions::Request> request,
      const std::shared_ptr<region_detection_msgs::srv::ShowSelectableRegions::Response> response)
  {
    (void)request_header;
    (void)response;
    setRegions(request->selectable_regions, request->start_selected);
  }

  void
  getSelectedRegionsCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                             const std::shared_ptr<region_detection_msgs::srv::GetSelectedRegions::Request> request,
                             const std::shared_ptr<region_detection_msgs::srv::GetSelectedRegions::Response> response)
  {
    (void)request_header;
    (void)request;
    response->selected_regions_indices = getSelectedRegions();
  }

  void buttonClickCallback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback)
  {
    using namespace visualization_msgs::msg;
    if (feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::BUTTON_CLICK)
    {
      return;
    }

    visualization_msgs::msg::InteractiveMarker int_marker;
    if (!interactive_marker_server_.get(feedback->marker_name, int_marker))
    {
      RCLCPP_WARN(node_->get_logger(), "The marker with id %s was not found", feedback->client_id.c_str());
      return;
    }
    bool selected = int_marker.controls.front().markers.front().action == Marker::ADD;

    int_marker.controls.front().markers.front().action = !selected ? Marker::ADD : Marker::DELETE;
    std_msgs::msg::ColorRGBA rgba_msg;
    std::tie(rgba_msg.r, rgba_msg.g, rgba_msg.b, rgba_msg.a) =
        !selected ? selection_colors_rgba::SELECTED : selection_colors_rgba::UNSELECTED;
    int_marker.controls.front().markers.front().color = rgba_msg;

    interactive_marker_server_.insert(int_marker);
    interactive_marker_server_.applyChanges();
  }

  void setRegions(const std::vector<geometry_msgs::msg::PoseArray>& regions, bool selected)
  {
    using namespace Eigen;
    using namespace interactive_markers;

    // clear all interactive markers
    interactive_marker_server_.clear();
    interactive_marker_server_.applyChanges();
    if(regions.empty())
    {
      return;
    }

    // create template marker
    visualization_msgs::msg::Marker marker;
    marker.scale.x = marker.scale.y = marker.scale.z = 1;
    marker.type = marker.TRIANGLE_LIST;
    marker.action = selected ? marker.ADD : marker.DELETE;  // marking selection flag
    marker.header.frame_id =
        regions.front().header.frame_id.empty() ? DEFAULT_FRAME_ID : regions.front().header.frame_id;

    std::tie(marker.color.r, marker.color.g, marker.color.b, marker.color.a) =
        selected ? selection_colors_rgba::SELECTED : selection_colors_rgba::UNSELECTED;

    InteractiveMarkerServer::FeedbackCallback button_callback_ = InteractiveMarkerServer::FeedbackCallback(
        boost::bind(&InteractiveRegionManager::buttonClickCallback, this, _1));

    // creating triangles
    geometry_msgs::msg::Point p1, p2, p3, p4;
    for (std::size_t i = 0; i < regions.size(); i++)
    {
      RCLCPP_INFO(node_->get_logger(), "Adding region %i with height %f", i, region_height_);
      marker.points.clear();
      geometry_msgs::msg::PoseArray region = regions[i];
      for (std::size_t j = 1; j < region.poses.size(); j++)
      {
        // converting to eigen pose
        Eigen::Affine3d pose1, pose2;
        tf2::fromMsg(region.poses[j - 1], pose1);
        tf2::fromMsg(region.poses[j], pose2);

        // computing points
        Vector3d eig_point;
        eig_point = pose1 * (0.5 * region_height_ * Vector3d::UnitZ());
        p1 = tf2::toMsg(eig_point);

        eig_point = pose2 * (0.5 * region_height_ * Vector3d::UnitZ());
        p2 = tf2::toMsg(eig_point);

        eig_point = pose2 * (-0.5 * region_height_ * Vector3d::UnitZ());
        p3 = tf2::toMsg(eig_point);

        eig_point = pose1 * (-0.5 * region_height_ * Vector3d::UnitZ());
        p4 = tf2::toMsg(eig_point);

        // creating triangles
        std::vector<geometry_msgs::msg::Point> triangle_points = { p1, p2, p3, p3, p4, p1 };
        std::for_each(triangle_points.begin(), triangle_points.end(), [&marker](const geometry_msgs::msg::Point& p) {
          marker.points.push_back(p);
        });

        triangle_points = { p1, p4, p3, p3, p2, p1 };
        std::for_each(triangle_points.begin(), triangle_points.end(), [&marker](const geometry_msgs::msg::Point& p) {
          marker.points.push_back(p);
        });
      }

      // creating region interactive marker now
      visualization_msgs::msg::InteractiveMarker int_marker;
      int_marker.name = REGION_ID_PREFIX + std::to_string(i);
      int_marker.pose = tf2::toMsg(Affine3d::Identity());

      // create button control
      visualization_msgs::msg::InteractiveMarkerControl button_control;
      button_control.interaction_mode = button_control.BUTTON;
      button_control.markers.push_back(marker);
      button_control.name = "button_" + int_marker.name;
      button_control.always_visible = true;

      // fill interactive marker
      int_marker.controls.push_back(button_control);
      int_marker.scale = 1;
      int_marker.header.frame_id = marker.header.frame_id;
      int_marker.description = int_marker.name;

      // add to server
      interactive_marker_server_.insert(int_marker, button_callback_);
    }

    // apply changes
    interactive_marker_server_.applyChanges();
  }

  std::vector<int> getSelectedRegions()
  {
    using namespace interactive_markers;
    using namespace visualization_msgs::msg;

    std::vector<int> selected_indices;
    for (std::size_t i = 0; i < interactive_marker_server_.size(); i++)
    {
      std::string id = REGION_ID_PREFIX + std::to_string(i);
      InteractiveMarker int_marker;
      interactive_marker_server_.get(id, int_marker);
      bool selected = int_marker.controls.front().markers.front().action == Marker::ADD;
      if (selected)
      {
        selected_indices.push_back(i);
      }
    }
    return selected_indices;
  }

  // rclcpp
  std::shared_ptr<rclcpp::Node> node_;
  interactive_markers::InteractiveMarkerServer interactive_marker_server_;
  rclcpp::Service<region_detection_msgs::srv::ShowSelectableRegions>::SharedPtr show_selectable_regions_server_;
  rclcpp::Service<region_detection_msgs::srv::GetSelectedRegions>::SharedPtr get_selected_regions_server_;

  // parameters
  double region_height_;
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
  std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("interactive_region_selection", options);
  InteractiveRegionManager mngr(node);
  rclcpp::spin(node);
  return 0;
}
