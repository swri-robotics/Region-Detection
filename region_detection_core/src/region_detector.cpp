/*
 * region_detector.cpp
 *
 *  Created on: Jun 4, 2020
 *      Author: jnicho
 */

#include <yaml-cpp/yaml.h>

#include <opencv2/highgui.hpp>

#include <boost/format.hpp>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>

#include "region_detector.h"

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
}

std::tuple<bool,std::string> RegionDetector::run2DAnalysis(const RegionDetectionConfig::OpenCVCfg& config, cv::Mat input,
                                                           std::vector<std::vector<cv::Point> > contours) const
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
      return std::make_tuple(success, err_msg);
    }

    // select dilation type
    if(DILATION_TYPES.count(config.dilation.elem) == 0)
    {
      success = false;
      err_msg = "invalid dilation element";
      LOG4CXX_INFO(logger_,err_msg);
      return std::make_tuple(success, err_msg);
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
  cv::findContours(output.clone(), contours, hierarchy, config.contour.mode, config.contour.method);

  cv::Mat drawing = cv::Mat::zeros( output.size(), CV_8UC3 );
  LOG4CXX_INFO(logger_,"Contour analysis found "<< contours.size() << "contours");
  for( int i = 0; i< contours.size(); i++ )
  {
   cv::Scalar color = cv::Scalar( RANDOM_NUM_GEN.uniform(0, 255), RANDOM_NUM_GEN.uniform(0,255), RANDOM_NUM_GEN.uniform(0,255) );
   double area = cv::contourArea(contours[i]);
   double arc_length = cv::arcLength(contours[i], false);
   cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
   std::string contour_summary = boost::str(boost::format("c[%i]: s: %i, area: %f, arc %f; (p0: %i, pf: %i); h: %i") %
                                    i % contours[i].size() % area % arc_length %
                                    contours[i].front() % contours[i].back() % hierarchy[i]);
   LOG4CXX_INFO(logger_,contour_summary);
   updateDebugWindow(output);
  }
  output = drawing.clone();
  return std::make_tuple(success, err_msg);
}

RegionDetector::RegionResults RegionDetector::compute(const std::vector<DataBundle> &input)
{
  return RegionResults();
}


} /* namespace region_detection_core */
