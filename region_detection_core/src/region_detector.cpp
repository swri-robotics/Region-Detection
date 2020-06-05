/*
 * region_detector.cpp
 *
 *  Created on: Jun 4, 2020
 *      Author: jnicho
 */

#include <yaml-cpp/yaml.h>

#include <boost/format.hpp>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>

#include "region_detector.h"

log4cxx::LoggerPtr createConsoleLogger(const std::string& logger_name)
{
  using namespace log4cxx;
  PatternLayoutPtr pattern_layout(new PatternLayout("[\%-5p] [\%c](L:\%L): \%m\%n"));
  ConsoleAppenderPtr console_appender(new ConsoleAppender(pattern_layout));
  log4cxx::LoggerPtr logger(Logger::getLogger(logger_name));
  logger->addAppender(console_appender);
  logger->setLevel(Level::getInfo());
  return logger;
}

static log4cxx::LoggerPtr DEFAULT_LOGGER = createConsoleLogger("Default");

namespace region_detection_core
{

RegionDetector::RegionDetector(log4cxx::LoggerPtr logger):
    logger_(logger ? logger : DEFAULT_LOGGER)
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
    logger_(logger ? logger : DEFAULT_LOGGER)
{
  if(!configure(config))
  {
    throw std::runtime_error("Invalid configuration");
  }
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

std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > RegionDetector::compute(
    const std::vector<DataBundle> &input)
{

  return {};
}

} /* namespace region_detection_core */
