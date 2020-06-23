/*
 * @author Jorge Nicho
 * @file region_detector.cpp
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

#include <yaml-cpp/yaml.h>

#include <opencv2/highgui.hpp>

#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

#include "region_detection_core/region_detector.h"

static const std::map<int,int> DILATION_TYPES = {{0, cv::MORPH_RECT}, {1, cv::MORPH_CROSS},{2, cv::MORPH_CROSS}};
static cv::RNG RANDOM_NUM_GEN(12345);

log4cxx::LoggerPtr createDefaultLogger(const std::string& logger_name)
{
  using namespace log4cxx;
  PatternLayoutPtr pattern_layout(new PatternLayout("[\%-5p] [\%c](L:\%L): \%m\%n"));
  ConsoleAppenderPtr console_appender(new ConsoleAppender(pattern_layout));
  log4cxx::LoggerPtr logger(Logger::getLogger(logger_name));
  logger->addAppender(console_appender);
  logger->setLevel(Level::getInfo());
  return logger;
}

Eigen::Matrix3d toRotationMatrix(const Eigen::Vector3d& vx, const Eigen::Vector3d& vy,
                                                    const Eigen::Vector3d& vz)
{
  using namespace Eigen;
  Matrix3d rot;
  rot.block(0,0,1,3) = Vector3d(vx.x(), vy.x(), vz.x()).array().transpose();
  rot.block(1,0,1,3) = Vector3d(vx.y(), vy.y(), vz.y()).array().transpose();
  rot.block(2,0,1,3) = Vector3d(vx.z(), vy.z(), vz.z()).array().transpose();
  return rot;
}

namespace region_detection_core
{

RegionDetector::RegionDetector(log4cxx::LoggerPtr logger):
    logger_(logger ? logger : createDefaultInfoLogger("Default"))
{
  RegionDetectionConfig cfg;
  if(!configure(cfg))
  {
    throw std::runtime_error("Invalid configuration");
  }
}

RegionDetector::~RegionDetector()
{

}

RegionDetector::RegionDetector(const RegionDetectionConfig &config, log4cxx::LoggerPtr logger):
    logger_(logger ? logger : createDefaultInfoLogger("Default"))
{
  if(!configure(config))
  {
    throw std::runtime_error("Invalid configuration");
  }
}

log4cxx::LoggerPtr RegionDetector::createDefaultInfoLogger(const std::string& logger_name)
{
  using namespace log4cxx;
  PatternLayoutPtr pattern_layout(new PatternLayout("[\%-5p] [\%c](L:\%L): \%m\%n"));
  ConsoleAppenderPtr console_appender(new ConsoleAppender(pattern_layout));
  log4cxx::LoggerPtr logger(Logger::getLogger(logger_name));
  logger->addAppender(console_appender);
  logger->setLevel(Level::getInfo());
  return logger;
}

log4cxx::LoggerPtr RegionDetector::createDefaultDebugLogger(const std::string& logger_name)
{
  using namespace log4cxx;
  PatternLayoutPtr pattern_layout(new PatternLayout("[\%-5p] [\%c](L:\%L): \%m\%n"));
  ConsoleAppenderPtr console_appender(new ConsoleAppender(pattern_layout));
  log4cxx::LoggerPtr logger(Logger::getLogger(logger_name));
  logger->addAppender(console_appender);
  logger->setLevel(Level::getDebug());
  return logger;
}

bool RegionDetector::configure(const RegionDetectionConfig &config)
{
  cfg_ = std::make_shared<RegionDetectionConfig>(config);
  return cfg_!=nullptr;
}

bool RegionDetector::configure(const std::string &yaml_str)
{
  return false;
}

log4cxx::LoggerPtr RegionDetector::getLogger()
{
  return logger_;
}

const RegionDetectionConfig& RegionDetector::getConfig()
{
  return *cfg_;
}

void RegionDetector::updateDebugWindow(const cv::Mat& im) const
{
  using namespace cv;
  const RegionDetectionConfig::OpenCVCfg& opencv_cfg = cfg_->opencv_cfg;

  if(!opencv_cfg.debug_mode_enable)
  {
    return;
  }

  // check if window is open
  const std::string wname = opencv_cfg.debug_window_name;
  if(cv::getWindowProperty(wname, cv::WND_PROP_VISIBLE) <= 0)
  {
    // create window then
    namedWindow(opencv_cfg.debug_window_name, WINDOW_AUTOSIZE);
  }

  imshow( wname, im );

  if(opencv_cfg.debug_wait_key)
  {
    cv::waitKey();
  }
}

RegionDetector::Result RegionDetector::compute2dContours(const RegionDetectionConfig::OpenCVCfg& config, cv::Mat input,
                                                           std::vector<std::vector<cv::Point> >& contours_indices) const
{
  bool success = false;
  std::string err_msg;
  cv::Mat output = input;

  //  ======================== convert to grayscale ========================
  cvtColor( output.clone(), output, cv::COLOR_RGB2GRAY );
  updateDebugWindow(output);

  // ======================== inverting ========================
  if(config.invert_image)
  {
    cv::Mat inverted = cv::Scalar_<uint8_t>(255) - input;
    output = inverted.clone();
    updateDebugWindow(output);
  }

  //  ======================== dilating ========================
  if(config.dilation.enable)
  {
    // check values
    if(config.dilation.size <= 0)
    {
      success = false;
      err_msg = "invalid dilation size";
      LOG4CXX_INFO(logger_,err_msg);
      return logAndReturn(success, err_msg);
    }

    // select dilation type
    if(DILATION_TYPES.count(config.dilation.elem) == 0)
    {
      success = false;
      err_msg = "invalid dilation element";
      LOG4CXX_INFO(logger_,err_msg);
      return logAndReturn(success, err_msg);
    }
    int dilation_type = DILATION_TYPES.at(config.dilation.elem);
    cv::Mat element = cv::getStructuringElement( dilation_type,
                         cv::Size( 2*config.dilation.size + 1, 2*config.dilation.size+1 ),
                         cv::Point( config.dilation.size, config.dilation.size ) );
    cv::dilate( output.clone(), output, element );
    updateDebugWindow(output);
  }

  //  ======================== Canny Edge Detection ========================
  if(config.canny.enable)
  {
    decltype(output) detected_edges;
    int aperture_size = 2 * config.canny.aperture_size + 1;
    aperture_size = aperture_size < 3  ? 3 : aperture_size;
    cv::Canny( output.clone(), detected_edges, config.canny.lower_threshold, config.canny.upper_threshold, aperture_size );
    output = detected_edges.clone();
    updateDebugWindow(output);
  }

  //  ======================== Contour Detection ========================
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(output.clone(), contours_indices, hierarchy, config.contour.mode, config.contour.method);

  cv::Mat drawing = cv::Mat::zeros( output.size(), CV_8UC3 );
  LOG4CXX_INFO(logger_,"Contour analysis found "<< contours_indices.size() << "contours");
  for( int i = 0; i< contours_indices.size(); i++ )
  {
   cv::Scalar color = cv::Scalar( RANDOM_NUM_GEN.uniform(0, 255), RANDOM_NUM_GEN.uniform(0,255), RANDOM_NUM_GEN.uniform(0,255) );
   double area = cv::contourArea(contours_indices[i]);
   double arc_length = cv::arcLength(contours_indices[i], false);
   cv::drawContours( drawing, contours_indices, i, color, 2, 8, hierarchy, 0, cv::Point() );
   std::string contour_summary = boost::str(boost::format("c[%i]: s: %i, area: %f, arc %f; (p0: %i, pf: %i); h: %i") %
                                    i % contours_indices[i].size() % area % arc_length %
                                    contours_indices[i].front() % contours_indices[i].back() % hierarchy[i]);
   LOG4CXX_INFO(logger_,contour_summary);
   updateDebugWindow(output);
  }

  // removing repeated points
  for(std::size_t i = 0; i < contours_indices.size() ;i++)
  {
    auto& cindices = contours_indices[i];
    std::sort(cindices.begin(), cindices.end(),[](cv::Point& p1, cv::Point& p2){
      return (p1.x + p1.y) > (p2.x + p2.y);
    });
    std::remove_reference< decltype(cindices) >::type::iterator last_iter = std::unique(
        cindices.begin(), cindices.end(),[](cv::Point& p1, cv::Point& p2){
      return (p1.x == p2.x) && (p1.y == p2.y);
    });
    cindices.erase(last_iter, cindices.end());
    contours_indices[i] = cindices;
  }

  output = drawing.clone();
  return logAndReturn(success, err_msg);
}

bool RegionDetector::compute(const std::vector<DataBundle> &input,
                                                      RegionDetector::RegionResults& regions)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> all_closed_contours_points;
  pcl::PointCloud<pcl::PointNormal>::Ptr normals = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();

  Result res;
  for(const DataBundle& data : input)
  {
    std::vector<std::vector<cv::Point> > contours_indices;
    res = compute2dContours(cfg_->opencv_cfg, data.image, contours_indices);
    if(!res)
    {
      return false;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> contours_points;
    res = extractContoursFromCloud(contours_indices,data.cloud,contours_points);
    if(!res)
    {
      return false;
    }

    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> contours_point_normals;
    res = computeNormals(data.cloud,contours_points,contours_point_normals);
    if(!res)
    {
      return false;
    }

    // adding found curves
    all_closed_contours_points.insert(all_closed_contours_points.end(),contours_points.begin(),
                                      contours_points.end());

    // adding point normals
    for(auto& cn : contours_point_normals)
    {
      (*normals) += *cn;
    }
  }

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> closed_curves_points;
  res = computeClosedRegions(all_closed_contours_points, closed_curves_points);
  if(!res)
  {
    return false;
  }

  res = computeRegionPoses(normals,closed_curves_points,regions);
  if(!res)
  {
    return false;
  }

  return true;
}

RegionDetector::Result RegionDetector::extractContoursFromCloud(
    const std::vector<std::vector<cv::Point> >& contour_indices, pcl::PointCloud<pcl::PointXYZ>::ConstPtr input,
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points)
{
  // check for organized point clouds
  if(input->isOrganized())
  {
    return logAndReturn(false,"Point Cloud not organized");
  }

  pcl::PointCloud<pcl::PointXYZ> temp_contour_points;
  for(const std::vector<cv::Point>& indices : contour_indices)
  {
    if(indices.empty())
    {
      return logAndReturn(false,"Empty indices vector was passed");
    }

    temp_contour_points.clear();
    for(const cv::Point& idx : indices)
    {
      if(idx.x >= input->width || idx.y >= input->height)
      {
        return logAndReturn(false,"2D indices exceed point cloud size");
      }
      temp_contour_points.push_back((*input)(idx.x,idx.y));
    }
    contours_points.push_back(boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(temp_contour_points));
  }

  return Result(!contours_points.empty(),"");
}

RegionDetector::Result RegionDetector::computeClosedRegions(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& contours_points,
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& closed_curves)
{
  using namespace pcl;
  // applying downsampling of each point cloud

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> output_contours_points;
  output_contours_points.assign(contours_points.begin(),contours_points.end());

  // downsampling
  if(cfg_->pcl_cfg.downsample.enable)
  {
    VoxelGrid<PointXYZ> voxelizer;
    double lf = cfg_->pcl_cfg.downsample.voxel_leafsize;
    voxelizer.setLeafSize(lf, lf, lf);
    for(std::size_t i = 0; i < output_contours_points.size(); i++)
    {
      PointCloud<PointXYZ>::Ptr downsampled_cloud = boost::make_shared<PointCloud<PointXYZ>>();
      voxelizer.setInputCloud(output_contours_points[i]);
      voxelizer.filter(*downsampled_cloud);
      output_contours_points[i] = downsampled_cloud;
    }
  }

  // order points
  for(std::size_t i = 0; i < output_contours_points.size(); i++)
  {
    if(!reorder(output_contours_points[i]->makeShared(),*output_contours_points[i]))
    {
      return logAndReturn(false,"Failed to order point cloud");
    }
  }

  std::vector<int> merged_curves_indices;
  // find closed curves
  for(std::size_t i = 0; i < output_contours_points.size() - 1; i++)
  {
    if(std::find(merged_curves_indices.begin(), merged_curves_indices.end(), i) != merged_curves_indices.end())
    {
      // already merged
      continue;
    }

    // get curve
    PointCloud<PointXYZ>::Ptr curve_points = output_contours_points[i]->makeShared();

    // create merge candidate index list
    std::vector<int> merge_candidate_indices;
    merge_candidate_indices.resize(output_contours_points.size());
    std::iota(merge_candidate_indices.begin(), merge_candidate_indices.end(), 0);

    // remove current curve
    merge_candidate_indices.erase(merge_candidate_indices.begin() + i);

    // search for close curves to merge with
    bool merged_curves = false;
    do
    {
      merged_curves = false;

      // merge with other unmerged curves
      for(int idx : merge_candidate_indices)
      {
        if(std::find(merged_curves_indices.begin(), merged_curves_indices.end(), idx) != merged_curves_indices.end())
        {
          // already merged
          continue;
        }

        PointCloud<PointXYZ>::Ptr merged_points = boost::make_shared<PointCloud<PointXYZ>>();
        PointCloud<PointXYZ>::Ptr next_curve_points = output_contours_points[idx];
        if(mergeCurves(*curve_points, *next_curve_points,*merged_points))
        {
          *curve_points = *merged_points;
          merged_curves_indices.push_back(i);
          merged_curves_indices.push_back(idx);
          merged_curves = true;
        }

        // removing repeated
        std::sort(merged_curves_indices.begin(), merged_curves_indices.end());
        auto it = std::unique(merged_curves_indices.begin(), merged_curves_indices.end());
        merged_curves_indices.resize( std::distance(merged_curves_indices.begin(),it) );
      }

    }while(merged_curves);

    // check if closed
    Eigen::Vector3d diff = (curve_points->front().getArray3fMap() - curve_points->back().getArray3fMap()).cast<double>();
    if(diff.norm() < cfg_->pcl_cfg.max_merge_dist)
    {
      // copying first point to end of cloud
      curve_points->push_back(curve_points->front());

      // saving
      closed_curves.push_back(curve_points);
      LOG4CXX_INFO(logger_,"Found closed curve with "<< curve_points->size() << " points");
      continue;
    }
  }

  if(closed_curves.empty())
  {
    return logAndReturn(false,"Found no closed curves");
  }

  return true;
}

RegionDetector::Result RegionDetector::mergeCurves(pcl::PointCloud<pcl::PointXYZ> c1, pcl::PointCloud<pcl::PointXYZ> c2,
                                 pcl::PointCloud<pcl::PointXYZ>& merged)
{
  std::vector<double> end_points_distances(4);

  // compute end point distances
  Eigen::Vector3d dist;
  dist = (c1.front().getArray3fMap() - c2.front().getArray3fMap()).cast<double>();
  end_points_distances[0] = dist.norm();

  dist = (c1.front().getArray3fMap() - c2.back().getArray3fMap()).cast<double>();
  end_points_distances[1] = dist.norm();

  dist = (c1.back().getArray3fMap() - c2.front().getArray3fMap()).cast<double>();
  end_points_distances[2] = dist.norm();

  dist = (c1.back().getArray3fMap() - c2.back().getArray3fMap()).cast<double>();
  end_points_distances[3] = dist.norm();

  std::vector<double>::iterator min_pos = std::min_element(end_points_distances.begin(), end_points_distances.end());
  if(*min_pos > cfg_->pcl_cfg.max_merge_dist)
  {
    // curves are too far, not merging
    return false;
  }

  std::size_t merge_method = std::distance(end_points_distances.begin(), min_pos);
  switch(merge_method)
  {
    case 0: // front2 to front1
      std::reverse(c2.begin(), c2.end());
      c1.insert(c1.begin(),c2.begin(), c2.end());
      break;

    case 1: // back2 to front1
      c1.insert(c1.begin(),c2.begin(), c2.end());
      break;

    case 2: // back1 to front2
      c1.insert(c1.end(),c2.begin(), c2.end());
      break;

    case 3: // back1 to back2
      std::reverse(c2.begin(), c2.end());
      c1.insert(c1.end(),c2.begin(), c2.end());
      break;
  }

  // reorder just in case
  return reorder(c1.makeShared(),merged);
}

RegionDetector::Result RegionDetector::logAndReturn(bool success, const std::string &err_msg) const
{
  if(success && !err_msg.empty())
  {
    LOG4CXX_INFO(logger_, err_msg);
  }
  else
  {
    LOG4CXX_ERROR(logger_, err_msg);
  }
  return Result(success,err_msg);
}

RegionDetector::Result RegionDetector::computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_cloud,
                                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &curves_points,
                                      std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> &curves_normals)
{

  // first compute normals
  pcl::PointCloud<pcl::PointNormal>::Ptr source_cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
  const config_3d::NormalEstimationCfg& cfg = cfg_->pcl_cfg.normal_est;
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
  ne.setInputCloud(source_cloud);
  ne.setViewPoint(cfg.viewpoint_xyz[0], cfg.viewpoint_xyz[1], cfg.viewpoint_xyz[2]);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (cfg.radius_search);
  ne.compute (*source_cloud_normals);

  // create kdtree to search cloud with normals
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setEpsilon(cfg.kdtree_epsilon);
  kdtree.setInputCloud(source_cloud);


  const int MAX_NUM_POINTS = 1;
  std::vector<int> nearest_indices(MAX_NUM_POINTS);
  std::vector<float> nearest_distances(MAX_NUM_POINTS);
  for(auto& curve : curves_points)
  {
    // search point and copy its normal
    pcl::PointCloud<pcl::PointNormal>::Ptr curve_normals = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    curve_normals->reserve(curve->size());
    for(auto& search_p : *curve)
    {
      int nearest_found = kdtree.radiusSearch(search_p,cfg.radius_search,nearest_indices,nearest_distances
                                              , MAX_NUM_POINTS);
      if(nearest_found <=0 )
      {
        return Result(false,"Found no points near curve, can not get normal vector");
      }
      curve_normals->push_back(source_cloud_normals->at(nearest_indices.front()));
    }

    curves_normals.push_back(curve_normals);
  }
  return true;
}

RegionDetector::Result RegionDetector::computeRegionPoses(pcl::PointCloud<pcl::PointNormal>::ConstPtr source_normal_cloud,
                                                      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &curves_points,
                                                      RegionResults &regions)
{
  using namespace Eigen;
  const config_3d::NormalEstimationCfg& cfg = cfg_->pcl_cfg.normal_est;

/*  // first compute normals
  pcl::PointCloud<pcl::Normal>::Ptr source_cloud_normals (new pcl::PointCloud<pcl::Normal>);

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(source_normal_cloud);
  ne.setViewPoint(cfg.viewpoint_xyz[0], cfg.viewpoint_xyz[1], cfg.viewpoint_xyz[2]);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (cfg.radius_search);
  ne.compute (*source_cloud_normals);*/

  // create kdtree to search cloud with normals
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_points = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::copyPointCloud(*source_normal_cloud, *source_points);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setEpsilon(cfg.kdtree_epsilon);
  kdtree.setInputCloud(source_points);

  const unsigned int MAX_NUM_POINTS = 1;
  std::vector<int> nearest_indices(MAX_NUM_POINTS);
  std::vector<float> nearest_distances(MAX_NUM_POINTS);
  for(auto& curve : curves_points)
  {
    // search point and copy its normal
    pcl::PointCloud<pcl::Normal>::Ptr curve_normals = boost::make_shared<pcl::PointCloud<pcl::Normal>>();
    curve_normals->reserve(curve->size());
    for(auto& search_p : *curve)
    {
      int nearest_found = kdtree.radiusSearch(search_p,cfg.radius_search,nearest_indices,nearest_distances
                                              , MAX_NUM_POINTS);
      if(nearest_found <=0 )
      {
        return Result(false,"Found no points near curve, can not get normal vector");
      }
      pcl::Normal p;
      pcl::copyPoint(source_normal_cloud->at(nearest_indices.front()), p);
      curve_normals->push_back(p);
    }

    EigenPose3dVector curve_poses;
    pcl::PointNormal p1, p2;
    Isometry3d pose;
    Vector3d x_dir, z_dir, y_dir;
    double dir = 1.0;
    for(std::size_t i = 0; i < curve->size(); i++)
    {
      std::size_t idx_current = i;
      std::size_t idx_next = i+1;
      dir = 1.0;

      if(idx_next >= curve->size())
      {
        std::size_t idx_current = i;
        std::size_t idx_next = i-1;
        dir = -1.0;
      }

      pcl::copyPoint(curve->at(idx_current),p1);
      pcl::copyPoint(curve_normals->at(idx_current),p1);
      pcl::copyPoint(curve->at(idx_next),p2);
      pcl::copyPoint(curve_normals->at(idx_next),p2);

      x_dir = dir * (p2.getVector3fMap() - p1.getVector3fMap()).normalized().cast<double>();
      z_dir = Vector3d(p1.normal_x, p1.normal_y, p1.normal_z).normalized();
      y_dir = z_dir.cross(x_dir).normalized();
      z_dir = x_dir.cross(y_dir).normalized();

      pose = Translation3d(p1.getVector3fMap().cast<double>());
      pose.matrix().block<3,3>(0,0) = toRotationMatrix(x_dir, y_dir, z_dir);
      curve_poses.push_back(pose);
    }
    regions.region_poses.push_back(curve_poses);
  }

  return true;
}

RegionDetector::Result RegionDetector::reorder(pcl::PointCloud<pcl::PointXYZ>::ConstPtr points,
                             pcl::PointCloud<pcl::PointXYZ>& ordered_points)
{
  using namespace pcl;
  const int k_points = 1;

  // build reordering kd tree
  pcl::KdTreeFLANN<pcl::PointXYZ> reorder_kdtree;
  reorder_kdtree.setEpsilon(cfg_->pcl_cfg.ordering.kdtree_epsilon);

  auto& cloud = *points;
  std::vector<int> visited_indices, active_indices;
  visited_indices.reserve(cloud.size());
  active_indices.resize(cloud.size());
  std::iota(active_indices.begin(), active_indices.end(),0);

  int search_point_idx = 0;
  visited_indices.push_back(search_point_idx); // insert first point

  const int max_iters = cloud.size();
  PointXYZ search_point = cloud[search_point_idx];
  PointXYZ start_point = search_point;
  PointXYZ closest_point;
  int iter_count = 0;

  // now reorder based on proximity
  while(iter_count <= max_iters)
  {
    iter_count++;

    // remove from active
    active_indices.erase(std::remove(active_indices.begin(), active_indices.end(),search_point_idx));

    if(active_indices.empty())
    {
      break;
    }

    // set tree inputs;
    IndicesConstPtr cloud_indices = boost::make_shared<const std::vector<int>>(active_indices);
    reorder_kdtree.setInputCloud(points,cloud_indices);

    // find next point
    std::vector<int> k_indices(k_points);
    std::vector<float> k_sqr_distances(k_points);
    int points_found = reorder_kdtree.nearestKSearch(search_point,k_points, k_indices, k_sqr_distances);
    if(points_found < k_points)
    {
      return logAndReturn(false,"Nearest K Search failed to find a point during reordering stage");
    }

    // insert new point if it has not been visited
    if(std::find(visited_indices.begin(), visited_indices.end(),k_indices[0]) != visited_indices.end())
    {
      // there should be no more points to add
      LOG4CXX_WARN(logger_,"Found repeated point during reordering stage, should not happen but proceeding");
      continue;
    }

    // check if found point is further away than the start point
    closest_point = cloud[k_indices[0]];
    Eigen::Vector3d diff = (start_point.getArray3fMap() - closest_point.getArray3fMap()).cast<double>();
    if(diff.norm() < std::sqrt(k_sqr_distances[0]))
    {
      // reversing will allow adding point from the other end
      std::reverse(visited_indices.begin(),visited_indices.end());
      start_point = cloud[visited_indices[0]];
    }

    // set next search point variables
    search_point_idx = k_indices[0];
    search_point = closest_point;

    // add to visited
    visited_indices.push_back(k_indices[0]);
  }

  if(visited_indices.size() < max_iters)
  {
    std::string err_msg = boost::str(
        boost::format("Failed to include all points in the downsampled group, only got %lu out of %lu") %
        visited_indices.size() % max_iters);
    return logAndReturn(false, err_msg);
  }
  pcl::copyPointCloud(*points,visited_indices,ordered_points);
  return true;
}

} /* namespace region_detection_core */
