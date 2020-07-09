/*
 * @author Jorge Nicho
 * @file detect_regions_demo.cpp
 * @date Jun 23, 2020
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

#include <ros/ros.h>

#include <Eigen/Geometry>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <visualization_msgs/MarkerArray.h>

#include <eigen_conversions/eigen_msg.h>

#include <pcl/conversions.h>

#include "pcl_ros/point_cloud.h"

#include <region_detection_core/region_detector.h>

using namespace region_detection_core;

typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > EigenPose3dVector;

static const std::string REGIONS_MARKERS_TOPIC = "regions_results";
static const std::string INPUT_CLOUD_TOPIC = "regions_cloud_input";
static const std::string REFERENCE_FRAME_ID = "results_frame";

static geometry_msgs::Pose pose3DtoPoseMsg(const std::array<float, 6>& p)
{
  using namespace Eigen;
  geometry_msgs::Pose pose_msg;
  Eigen::Affine3d eigen_pose = Translation3d(Vector3d(p[0],p[1],p[2])) *
      AngleAxisd(p[3],Vector3d::UnitX()) * AngleAxisd(p[4],Vector3d::UnitY()) *
      AngleAxisd(p[5],Vector3d::UnitZ());

  tf::poseEigenToMsg(eigen_pose,pose_msg);
  return std::move(pose_msg);
}

visualization_msgs::MarkerArray convertToAxisMarkers(const std::vector<EigenPose3dVector>& path,
                                                     const std::string& frame_id, const std::string& ns,
                                                     const std::size_t& start_id, const double& axis_scale,
                                                     const double& axis_length,
                                                     const std::array<float, 6>& offset)
{
  using namespace Eigen;

  visualization_msgs::MarkerArray markers;

  auto create_line_marker = [&](const int id, const std::tuple<float, float, float, float>& rgba) -> visualization_msgs::Marker
  {
    visualization_msgs::Marker line_marker;
    line_marker.action = line_marker.ADD;
    std::tie(line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a) = rgba;
    line_marker.header.frame_id = frame_id;
    line_marker.type = line_marker.LINE_LIST;
    line_marker.id = id;
    line_marker.lifetime = ros::Duration(0);
    line_marker.ns = ns;
    std::tie(line_marker.scale.x, line_marker.scale.y, line_marker.scale.z) = std::make_tuple(axis_scale, 0.0, 0.0);
    line_marker.pose = pose3DtoPoseMsg(offset);
    return std::move(line_marker);
  };

  // markers for each axis line
  int marker_id = start_id;
  visualization_msgs::Marker x_axis_marker = create_line_marker(++marker_id,std::make_tuple(1.0, 0.0, 0.0, 1.0));
  visualization_msgs::Marker y_axis_marker = create_line_marker(++marker_id,std::make_tuple(0.0, 1.0, 0.0, 1.0));
  visualization_msgs::Marker z_axis_marker = create_line_marker(++marker_id,std::make_tuple(0.0, 0.0, 1.0, 1.0));

  auto add_axis_line = [](const Isometry3d& eigen_pose, const Vector3d& dir,
      const geometry_msgs::Point& p1, visualization_msgs::Marker& marker){

    geometry_msgs::Point p2;
    Eigen::Vector3d line_endpoint;

    // axis endpoint
    line_endpoint = eigen_pose * dir;
    std::tie(p2.x, p2.y, p2.z)  = std::make_tuple(line_endpoint.x(), line_endpoint.y(), line_endpoint.z());

    // adding line
    marker.points.push_back(p1);
    marker.points.push_back(p2);
  };

  for(auto& poses : path)
  {
    for(auto& pose : poses)
    {
      geometry_msgs::Point p1;
      std::tie(p1.x, p1.y, p1.z) = std::make_tuple(pose.translation().x(), pose.translation().y(),
                                                   pose.translation().z());
      add_axis_line(pose,Vector3d::UnitX()* axis_length,p1, x_axis_marker);
      add_axis_line(pose,Vector3d::UnitY()* axis_length,p1, y_axis_marker);
      add_axis_line(pose,Vector3d::UnitZ()* axis_length,p1, z_axis_marker);
    }
  }

  markers.markers.push_back(x_axis_marker);
  markers.markers.push_back(y_axis_marker);
  markers.markers.push_back(z_axis_marker);
  return std::move(markers);
}

visualization_msgs::MarkerArray convertToDottedLineMarker(const std::vector<EigenPose3dVector>& path,
                                                    const std::string& frame_id,
                                                    const std::string& ns,
                                                    const std::size_t& start_id = 0,
                                                    const std::array<double,4>& rgba_line = {1.0, 1.0, 0.2, 1.0},
                                                    const std::array<double,4>& rgba_point = {0.1, .8, 0.2, 1.0},
                                                    const std::array<float, 6>& offset = {0,0,0,0,0,0},
                                                    const float& line_width = 0.001,
                                                    const float& point_size = 0.0015)
{
  visualization_msgs::MarkerArray markers_msgs;
  visualization_msgs::Marker line_marker, points_marker;

  // configure line marker
  line_marker.action = line_marker.ADD;
  std::tie(line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a) = std::make_tuple(rgba_line[0],
                                                                                                                 rgba_line[1],
                                                                                                                 rgba_line[2],
                                                                                                                 rgba_line[3]);
  line_marker.header.frame_id = frame_id;
  line_marker.type = line_marker.LINE_STRIP;
  line_marker.id = start_id;
  line_marker.lifetime = ros::Duration(0);
  line_marker.ns = ns;
  std::tie(line_marker.scale.x, line_marker.scale.y, line_marker.scale.z) = std::make_tuple(line_width, 0.0, 0.0);
  line_marker.pose = pose3DtoPoseMsg(offset);

  // configure point marker
  points_marker = line_marker;
  points_marker.type = points_marker.POINTS;
  points_marker.ns = ns;
  std::tie(points_marker.color.r, points_marker.color.g, points_marker.color.b, points_marker.color.a) = std::make_tuple(rgba_point[0],
                                                                                                                         rgba_point[1],
                                                                                                                         rgba_point[2],
                                                                                                                         rgba_point[3]);
  std::tie(points_marker.scale.x, points_marker.scale.y, points_marker.scale.z) = std::make_tuple(point_size,point_size,point_size);

  int id_counter = start_id;
  for(auto& poses : path)
  {
    line_marker.points.clear();
    points_marker.points.clear();
    line_marker.points.reserve(poses.size());
    points_marker.points.reserve(poses.size());
    for(auto& pose : poses)
    {
      geometry_msgs::Point p;
      std::tie(p.x, p.y, p.z) = std::make_tuple(pose.translation().x(), pose.translation().y(),
                                                pose.translation().z());
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

RegionDetectionConfig::OpenCVCfg loadOpenCVParams()
{
  ros::NodeHandle ph("~");
  bool success;
  std::string param_ns;
  RegionDetectionConfig::OpenCVCfg cfg;

  // top level parameters
  ph = ros::NodeHandle("~/config_opencv");
  success = ph.getParam("invert_image", cfg.invert_image) &&
    ph.getParam("debug_mode_enable", cfg.debug_mode_enable)&&
    ph.getParam("debug_window_name", cfg.debug_window_name)&&
    ph.getParam("debug_wait_key", cfg.debug_wait_key);
  if(!success)
  {
    std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % "top level");
    throw std::runtime_error(err_msg);
  }

  param_ns = "threshold";
  ph = ros::NodeHandle("~/config_opencv/" + param_ns);
  success =  ph.getParam("enable",cfg.threshold.enable) &&
     ph.getParam("value",cfg.threshold.value ) &&
     ph.getParam("type",cfg.threshold.type );
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  param_ns = "dilation";
  ph = ros::NodeHandle("~/config_opencv/" + param_ns);
  success =  ph.getParam("enable",cfg.dilation.enable) &&
     ph.getParam("elem",cfg.dilation.elem) &&
     ph.getParam("kernel_size",cfg.dilation.kernel_size);
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  param_ns = "canny";
  ph = ros::NodeHandle("~/config_opencv/" + param_ns);
  success =  ph.getParam("enable",cfg.canny.enable) &&
     ph.getParam("lower_threshold",cfg.canny.lower_threshold) &&
     ph.getParam("upper_threshold",cfg.canny.upper_threshold) &&
     ph.getParam("aperture_size",cfg.canny.aperture_size);
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  param_ns = "contour";
  ph = ros::NodeHandle("~/config_opencv/" + param_ns);
  success =  ph.getParam("mode",cfg.contour.mode) &&
     ph.getParam("method",cfg.contour.method);
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  return cfg;
}

RegionDetectionConfig::PCL2DCfg loadPCL2DParams()
{
  RegionDetectionConfig::PCL2DCfg cfg;
  ros::NodeHandle ph("~");
  bool success;
  std::string param_ns;

  ph = ros::NodeHandle("~/config_pcl2d");
  success =  ph.getParam("downsampling_radius",cfg.downsampling_radius) &&
     ph.getParam("split_dist",cfg.split_dist ) &&
     ph.getParam("closed_curve_max_dist",cfg.closed_curve_max_dist ) &&
     ph.getParam("simplification_min_points",cfg.simplification_min_points ) &&
     ph.getParam("simplification_alpha",cfg.simplification_alpha );
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % "config_pcl2d top level");
   throw std::runtime_error(err_msg);
  }
  return cfg;
}

RegionDetectionConfig::PCLCfg loadPCLParams()
{
  RegionDetectionConfig::PCLCfg cfg;
  ros::NodeHandle ph("~");
  bool success;
  std::string param_ns;

  // top level parameters
  ph = ros::NodeHandle("~/config_pcl");
  success =  ph.getParam("debug_mode_enable",cfg.debug_mode_enable) &&
     ph.getParam("max_merge_dist",cfg.max_merge_dist ) &&
     ph.getParam("closed_curve_max_dist",cfg.closed_curve_max_dist ) &&
     ph.getParam("simplification_min_dist",cfg.simplification_min_dist ) &&
     ph.getParam("min_num_points",cfg.min_num_points );
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % "top level");
   throw std::runtime_error(err_msg);
  }

  param_ns = "stat_removal";
  ph = ros::NodeHandle("~/config_pcl/" + param_ns);
  success =  ph.getParam("enable",cfg.stat_removal.enable) &&
      ph.getParam("kmeans",cfg.stat_removal.kmeans) &&
     ph.getParam("stddev",cfg.stat_removal.stddev );
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  param_ns = "normal_est";
  std::vector<double> viewpoint_vals(6);
  ph = ros::NodeHandle("~/config_pcl/" + param_ns);
  success =  ph.getParam("kdtree_epsilon",cfg.normal_est.kdtree_epsilon) &&
      ph.getParam("search_radius",cfg.normal_est.search_radius) &&
      ph.getParam("viewpoint_xyz",viewpoint_vals) &&
      ph.getParam("downsampling_radius",cfg.normal_est.downsampling_radius) ;
  std::copy(viewpoint_vals.begin(), viewpoint_vals.end(), cfg.normal_est.viewpoint_xyz.begin());
  if(!success)
  {
   std::string err_msg = boost::str(boost::format("Failed to load \"%s\" parameters") % param_ns);
   throw std::runtime_error(err_msg);
  }

  return cfg;
}

std::vector<RegionDetector::DataBundle> loadData()
{
  using namespace XmlRpc;
  using namespace Eigen;

  namespace fs = boost::filesystem;
  std::vector<RegionDetector::DataBundle> data_vec;
  ros::NodeHandle ph("~");
  bool success;
  std::string param_ns = "data";

  // first data directory
  std::string data_dir;
  if(!ph.getParam("data_dir",data_dir) || !fs::exists(fs::path(data_dir)))
  {
    std::string err_msg = boost::str(boost::format("The data directory \"%s\" could not be found") % data_dir);
    throw std::runtime_error(err_msg);
  }
  fs::path data_dir_path(data_dir);

  XmlRpcValue data_entries;
  if(!ph.getParam(param_ns, data_entries))
  {
    std::string err_msg = boost::str(boost::format("Failed to load data entries from \"%s\" parameter") % param_ns);
    throw std::runtime_error(err_msg);
  }

  if(data_entries.getType() != data_entries.TypeArray)
  {
    std::string err_msg = boost::str(boost::format("The \"%s\" parameter is not an array") % param_ns);
    throw std::runtime_error(err_msg);
  }

  for(int i = 0; i < data_entries.size(); i++)
  {
    RegionDetector::DataBundle bundle;
    XmlRpcValue entry = data_entries[i];
    success = entry.hasMember("image_file") && entry.hasMember("cloud_file") && entry.hasMember("transform");
    if(!success)
    {
      std::string err_msg = boost::str(
          boost::format("One or more fields not found in entry of \"%s\" parameter") % param_ns);
      throw std::runtime_error(err_msg);
    }

    fs::path image_file_path = data_dir_path / fs::path(static_cast<std::string>(entry["image_file"]));
    fs::path cloud_file_path = data_dir_path / fs::path(static_cast<std::string>(entry["cloud_file"]));
    std::vector<fs::path> file_paths = {image_file_path, cloud_file_path};
    if(!std::all_of(file_paths.begin(), file_paths.end(),[](const fs::path& p){
      if(!fs::exists(p))
      {
        std::string err_msg = boost::str(
            boost::format("The file \"%s\" does not exists") % p.string());
        ROS_ERROR_STREAM(err_msg);
        return false;
      }
      return true;
    }))
    {
      throw std::runtime_error("File not found");
    }

    // load files now
    bundle.image = cv::imread(image_file_path.string(), cv::IMREAD_COLOR);
    pcl::io::loadPCDFile(cloud_file_path.string(),bundle.cloud_blob);
    std::vector<double> transform_vals;
    XmlRpcValue transform_entry = entry["transform"];
    for(int j = 0; j < transform_entry.size(); j++)
    {
      transform_vals.push_back(static_cast<double>(transform_entry[j]));
    }
    bundle.transform = Translation3d(Vector3d(transform_vals[0], transform_vals[1], transform_vals[2])) *
        AngleAxisd(transform_vals[3],Vector3d::UnitX()) *
        AngleAxisd(transform_vals[4],Vector3d::UnitY()) *
        AngleAxisd(transform_vals[5],Vector3d::UnitZ());

    data_vec.push_back(std::move(bundle));
  }

  return data_vec;
}

int main(int argc, char** argv)
{
  using PointType = pcl::PointXYZRGB;

  ros::init(argc, argv, "detect_regions_demo");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(2);
  spinner.start();

  // getting configuration parameters
  RegionDetectionConfig cfg;
  cfg.opencv_cfg = loadOpenCVParams();
  cfg.pcl_2d_cfg = loadPCL2DParams();
  cfg.pcl_cfg = loadPCLParams();

  ROS_INFO("Loaded configuration parameters");

  // loading data location parameters
  /**
   * data:
   *  - image_file: dir1/color.png
   *    cloud_file: dir1/cloud.pcd
   *    transform: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [px, py, pz, rx, ry, rz]
   *  - image_file: dir2/color.png
   *    cloud_file: dir2/cloud.pcd
   *    transform: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [px, py, pz, rx, ry, rz]
   */
  std::vector<RegionDetector::DataBundle> data_vec = loadData();

  // computing regions now
  RegionDetector rd(cfg,RegionDetector::createDefaultDebugLogger("RD_Debug"));
  RegionDetector::RegionResults results;
  if(!rd.compute(data_vec,results))
  {
    ROS_ERROR("Failed to compute regions");
    //return -1;
  }

  // creating publishers
  ros::Publisher results_markers_pub = nh.advertise<visualization_msgs::MarkerArray>(REGIONS_MARKERS_TOPIC,1);
  ros::Publisher input_cloud_pub = nh.advertise<pcl::PointCloud<PointType>>(INPUT_CLOUD_TOPIC,1);

  // create markers to publish
  visualization_msgs::MarkerArray results_markers;
  int id = 0;
  for(auto& poses : results.closed_regions_poses)
  {
    id++;
    visualization_msgs::MarkerArray m = convertToAxisMarkers({poses},REFERENCE_FRAME_ID,"closed_regions_axes" + std::to_string(id),0, 0.001,
                                                             0.01,{0,0,0,0,0,0});
    results_markers.markers.insert(results_markers.markers.end(), m.markers.begin(), m.markers.end());

    m = convertToDottedLineMarker({poses},REFERENCE_FRAME_ID,"closed_regions_lines" + std::to_string(id));
    results_markers.markers.insert(results_markers.markers.end(), m.markers.begin(), m.markers.end());
  }

  id = 0;
  for(auto& poses : results.open_regions_poses)
  {
    id++;
    visualization_msgs::MarkerArray m = convertToAxisMarkers({poses},REFERENCE_FRAME_ID,"open_regions_axes" + std::to_string(id),0, 0.001,
                                                             0.01,{0,0,0,0,0,0});
    results_markers.markers.insert(results_markers.markers.end(), m.markers.begin(), m.markers.end());

    m = convertToDottedLineMarker({poses},REFERENCE_FRAME_ID,"open_regions_lines" + std::to_string(id),
                                                                  0,
                                                                  {1.0, 0.2, 0.2, 1.0},
                                                                  {0.2, 0.2, 0.2, 1.0});
    results_markers.markers.insert(results_markers.markers.end(), m.markers.begin(), m.markers.end());
  }

  // creating point cloud to publish
  pcl::PointCloud<PointType> input_cloud;
  for(auto& data : data_vec)
  {
    pcl::PointCloud<PointType>::Ptr color_cloud = boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromPCLPointCloud2(data.cloud_blob,*color_cloud);
    input_cloud += (*color_cloud);
  }
  input_cloud.header.frame_id = REFERENCE_FRAME_ID;


  ros::Duration loop_pause(0.2);
  while(ros::ok())
  {
    results_markers_pub.publish(results_markers);
    input_cloud_pub.publish(input_cloud);
    loop_pause.sleep();
  }

  ros::waitForShutdown();
  return 0;
}




