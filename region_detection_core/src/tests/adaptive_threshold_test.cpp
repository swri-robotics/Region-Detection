/**
 * @file Threshold.cpp
 * @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <boost/filesystem.hpp>
#include <iostream>

using namespace cv;
using std::cout;

/// Global variables

int threshold_value = 0;
int threshold_type = cv::ThresholdTypes::THRESH_BINARY;
int threshold_method = cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C;
int block_size = 1;

int const max_value = 255;
int const max_type = cv::ThresholdTypes::THRESH_BINARY_INV;
int const max_method = cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C;
int const max_block_size = 5;

Mat src, src_gray, dst, inverted;
const char* window_name = "Threshold Demo";

int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero "
                            "Inverted";
const char* trackbar_value = "Value";

//![updateImage]
/**
 * @function updateImage
 */
static void updateImage(int, void*)
{
  /* 0: Binary
   1: Binary Inverted
   2: Threshold Truncated
   3: Threshold to Zero
   4: Threshold to Zero Inverted
  */
  inverted = cv::Scalar_<uint8_t>(255) - src_gray;
  if (dilation_size > 0)
  {
    int dilation_type = 0;
    if (dilation_elem == 0)
    {
      dilation_type = MORPH_RECT;
    }
    else if (dilation_elem == 1)
    {
      dilation_type = MORPH_CROSS;
    }
    else if (dilation_elem == 2)
    {
      dilation_type = MORPH_ELLIPSE;
    }
    Mat element = getStructuringElement(
        dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
    decltype(inverted) dilation_dst;
    dilate(inverted, dilation_dst, element);
    dst = dilation_dst.clone();
  }
  else
  {
    std::cout << "Skipping dilation" << std::endl;
    dst = inverted.clone();
  }

  block_size = block_size == 0 ? 1 : block_size;
  // threshold_type = threshold_type == max_type ? cv::ThresholdTypes::THRESH_TRIANGLE :  threshold_type;

  int block_size_val = block_size * 2 + 1;
  adaptiveThreshold(dst.clone(), dst, threshold_value, threshold_method, threshold_type, block_size_val, 0);

  imshow(window_name, dst);
}
//![updateImage]

/**
 * @function main
 */
int main(int argc, char** argv)
{
  namespace fs = boost::filesystem;
  //! [load]
  String image_name;
  if (argc > 1)
  {
    image_name = argv[1];
  }

  if (!fs::exists(fs::path(image_name.c_str())))
  {
    std::cout << "Failed to load file " << image_name.c_str() << std::endl;
    return -1;
  }

  src = imread(image_name, IMREAD_COLOR);  // Load an image

  if (src.empty())
  {
    cout << "Cannot read the image: " << image_name << std::endl;
    return -1;
  }

  // cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
  cvtColor(src, src_gray, COLOR_RGB2GRAY);

  //! [load]

  //! [window]
  namedWindow(window_name, WINDOW_AUTOSIZE);  // Create a window to display results
  //! [window]

  //! [trackbar]
  createTrackbar(trackbar_value, window_name, &threshold_value, max_value, updateImage);  // Create a Trackbar to choose
                                                                                          // Threshold value

  createTrackbar(trackbar_type, window_name, &threshold_type, max_type, updateImage);  // Create a Trackbar to choose
                                                                                       // type of Threshold

  createTrackbar("Method", window_name, &threshold_method, max_method, updateImage);  // Create a Trackbar to choose
                                                                                      // Threshold value

  createTrackbar(
      "Block Size: 2 x (val) + 1", window_name, &block_size, max_block_size, updateImage);  // Create a Trackbar to
                                                                                            // choose Threshold value

  createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", window_name, &dilation_elem, max_elem, updateImage);
  createTrackbar("Kernel size:\n 2n +1", window_name, &dilation_size, max_kernel_size, updateImage);
  //! [trackbar]

  updateImage(0, 0);  // Call the function to initialize

  /// Wait until the user finishes the program
  waitKey();
  return 0;
}
/*
 * adaptive_threshold_test.cpp
 *
 *  Created on: Jun 3, 2020
 *      Author: jnicho
 */
