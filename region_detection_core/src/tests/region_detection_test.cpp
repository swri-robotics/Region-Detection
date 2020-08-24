
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include "region_detection_core/region_detector.h"

using namespace region_detection_core;

int main(int argc, char** argv)
{
  namespace fs = boost::filesystem;

  auto logger = RegionDetector::createDefaultDebugLogger("DEBUG");
  if (argc < 3)
  {
    LOG4CXX_ERROR(logger, "Needs config file and image arguments");
    return -1;
  }

  std::vector<std::string> files = { argv[1], argv[2] };
  if (!std::all_of(files.begin(), files.end(), [](const std::string& f) { return fs::exists(fs::path(f)); }))
  {
    LOG4CXX_ERROR(logger, "File does not exists");
    return -1;
  }

  bool compute_contours = false;
  if (argc == 4)
  {
    compute_contours = boost::lexical_cast<bool>(argv[3]);
  }

  std::string config_file = argv[1];
  std::string img_file = argv[2];

  cv::Mat output;
  RegionDetector region_detector(logger);
  int key;
  do
  {
    if (!region_detector.configureFromFile(config_file))
    {
      LOG4CXX_ERROR(logger, "Failed to load configuration from file " << config_file);
      return -1;
    }
    std::vector<std::vector<cv::Point> > contours_indices;
    cv::Mat input = cv::imread(img_file, cv::IMREAD_COLOR);  // Load an image
    if (compute_contours)
    {
      region_detector.compute2d(input, output, contours_indices);
    }
    else
    {
      region_detector.compute2d(input, output);
    }

    std::cout << "Pres ESC to exit" << std::endl;
    key = cv::waitKey();
  } while (key != 27);  // escape

  std::cout << "Key pressed " << key << std::endl;
  return 0;
}
