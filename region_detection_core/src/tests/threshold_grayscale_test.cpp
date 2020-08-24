/**
 * @file Threshold.cpp
 * @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iostream>

using namespace cv;
using std::cout;

/// Global variables

int threshold_enabled = 0;
int threshold_value = 150;
int threshold_type = cv::ThresholdTypes::THRESH_TRUNC;
int const max_value = 255;

int const max_threshold_enabled = 1;
int const max_type = cv::ThresholdTypes::THRESH_TOZERO_INV;
int const max_binary_value = 255;

int dilation_enabled = 1;
int dilation_elem = 0;
int dilation_size = 1;
int const max_elem = 2;
int const max_kernel_size = 21;

int thinning_enabled = 1;

struct CountourConfig
{
  int enable = 1;
  int mode = CV_RETR_EXTERNAL;
  int method = CV_CHAIN_APPROX_SIMPLE;

  static const int MAX_ENABLE = 1;
  static const int MAX_MODE = CV_RETR_TREE;
  static const int MAX_METHOD = CV_CHAIN_APPROX_TC89_KCOS;
};
static CountourConfig contour_cfg;

struct CannyConfig
{
  int enable = 1;
  int lower_threshold = 45;
  int upper_threshold = lower_threshold * 3;
  int aperture_size = 1;

  static const int MAX_ENABLE = 1;
  static const int MAX_LOWER_TH = 100;
  static const int MAX_UPPER_TH = 255;
  static const int MAX_APERTURE_SIZE = 3;
};
static CannyConfig canny_cfg;

static std::string RESULTS_DIR;
static int SAVE_IMAGE = 0;

cv::RNG rng(12345);

Mat src, src_gray, dst, inverted;
const char* window_name = "Countour detection";
const char* trackbar_type = "Threshold Type: \n 0: Binary, 1: Binary Inverted, 2: Truncate, 3: To Zero, 4: To Zero "
                            "Inverted\n";
const char* trackbar_value = "Threshold Value";

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningGuoHallIteration(cv::Mat& im, int iter)
{
  cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

  for (int i = 1; i < im.rows; i++)
  {
    for (int j = 1; j < im.cols; j++)
    {
      uchar p2 = im.at<uchar>(i - 1, j);
      uchar p3 = im.at<uchar>(i - 1, j + 1);
      uchar p4 = im.at<uchar>(i, j + 1);
      uchar p5 = im.at<uchar>(i + 1, j + 1);
      uchar p6 = im.at<uchar>(i + 1, j);
      uchar p7 = im.at<uchar>(i + 1, j - 1);
      uchar p8 = im.at<uchar>(i, j - 1);
      uchar p9 = im.at<uchar>(i - 1, j - 1);

      int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
      int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
      int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
      int N = N1 < N2 ? N1 : N2;
      int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

      if (C == 1 && (N >= 2 && N <= 3) & m == 0)
        marker.at<uchar>(i, j) = 1;
    }
  }

  im &= ~marker;
}

void thinningGuoHall(cv::Mat& im)
{
  im /= 255;

  cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
  cv::Mat diff;

  do
  {
    thinningGuoHallIteration(im, 0);
    thinningGuoHallIteration(im, 1);
    cv::absdiff(im, prev, diff);
    im.copyTo(prev);
  } while (cv::countNonZero(diff) > 0);

  im *= 255;
}

//![updateImage]
/**
 * @function updateImage
 */
static void updateImage(int, void*)
{
  namespace fs = boost::filesystem;
  /* 0: Binary
   1: Binary Inverted
   2: Threshold Truncated
   3: Threshold to Zero
   4: Threshold to Zero Inverted
  */
  inverted = cv::Scalar_<uint8_t>(255) - src_gray;
  dst = inverted.clone();

  if (dilation_enabled)
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
    dilate(inverted, dst, element);
  }
  else
  {
    std::cout << "Skipping Dilation" << std::endl;
  }

  // threshold( src_gray, dst, threshold_value, max_binary_value, threshold_type );
  if (threshold_enabled == 1)
  {
    threshold(dst.clone(), dst, threshold_value, max_binary_value, threshold_type);
  }
  else
  {
    std::cout << "Skipping Threshold" << std::endl;
  }

  // thining
  // thinningGuoHall(dst);

  // canny edge detection
  if (canny_cfg.enable == 1)
  {
    /// Canny detector
    decltype(dst) detected_edges;
    int aperture_size = 2 * canny_cfg.aperture_size + 1;
    aperture_size = aperture_size < 3 ? 3 : aperture_size;
    cv::Canny(dst.clone(), detected_edges, canny_cfg.lower_threshold, canny_cfg.upper_threshold, aperture_size);
    dst = detected_edges.clone();
  }
  else
  {
    std::cout << "Skipping Canny" << std::endl;
  }

  // thining
  if (thinning_enabled == 1)
  {
    thinningGuoHall(dst);
  }

  // contour
  if (contour_cfg.enable == 1)
  {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dst.clone(), contours, hierarchy, contour_cfg.mode, contour_cfg.method);

    Mat drawing = Mat::zeros(dst.size(), CV_8UC3);
    std::cout << "Contour found " << contours.size() << " contours" << std::endl;
    for (int i = 0; i < contours.size(); i++)
    {
      Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      double area = cv::contourArea(contours[i]);
      double arc_length = cv::arcLength(contours[i], false);
      drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
      std::string summary =
          boost::str(boost::format("c[%i]: s: %i, area: %f, arc %f; (p0: %i, pf: %i); h: %i") % i % contours[i].size() %
                     area % arc_length % contours[i].front() % contours[i].back() % hierarchy[i]);
      std::cout << summary << std::endl;
    }
    dst = drawing.clone();
  }
  else
  {
    std::cout << "Skipping Contour" << std::endl;
  }

  imshow(window_name, dst);

  if (SAVE_IMAGE == 1)
  {
    SAVE_IMAGE = 0;

    std::string save_path = (fs::path(RESULTS_DIR) / fs::path("results.png")).string();
    imwrite(save_path, dst);
    std::cout << "Saved image in path " << save_path << std::endl;
  }
}
//![Threshold_Demo]

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
  RESULTS_DIR = fs::path(image_name).parent_path().string();

  src = imread(image_name, IMREAD_COLOR);  // Load an image

  if (src.empty())
  {
    cout << "Cannot read the image: " << image_name << std::endl;
    return -1;
  }

  // cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
  cvtColor(src, src_gray, COLOR_RGB2GRAY);

  //! [window]
  namedWindow(window_name, WINDOW_AUTOSIZE);  // Create a window to display results

  //! [Threshold trackbars]
  createTrackbar("Threshold Enable",
                 window_name,
                 &threshold_enabled,
                 max_threshold_enabled,
                 updateImage);  // Create a Trackbar to choose type of Threshold

  createTrackbar(trackbar_type, window_name, &threshold_type, max_type, updateImage);  // Create a Trackbar to choose
                                                                                       // type of Threshold

  createTrackbar(trackbar_value, window_name, &threshold_value, max_value, updateImage);  // Create a Trackbar to choose
                                                                                          // Threshold value

  //! [Canny trackbar]
  createTrackbar("Canny Enable", window_name, &canny_cfg.enable, canny_cfg.MAX_ENABLE, updateImage);
  createTrackbar("Canny Lower Threshold", window_name, &canny_cfg.lower_threshold, canny_cfg.MAX_LOWER_TH, updateImage);
  createTrackbar("Canny Upper Threshold", window_name, &canny_cfg.upper_threshold, canny_cfg.MAX_UPPER_TH, updateImage);
  createTrackbar(
      "Canny Aperture Size 2n + 1", window_name, &canny_cfg.aperture_size, canny_cfg.MAX_APERTURE_SIZE, updateImage);

  //! [Thinning trackbar]
  createTrackbar("Thinning Enable", window_name, &thinning_enabled, 1, updateImage);

  //! [Dilation trackbars]
  createTrackbar("Dilation Enabled", window_name, &dilation_enabled, 1, updateImage);
  createTrackbar(
      "Dilation Element:\n 0: Rect , 1: Cross, 2: Ellipse", window_name, &dilation_elem, max_elem, updateImage);
  createTrackbar("Dilation Kernel size:\n 2n +1", window_name, &dilation_size, max_kernel_size, updateImage);

  //! [Contour trackbar]
  createTrackbar("Contour Enable", window_name, &contour_cfg.enable, contour_cfg.MAX_ENABLE, updateImage);
  createTrackbar("Contour Method", window_name, &contour_cfg.method, contour_cfg.MAX_METHOD, updateImage);
  createTrackbar("Contour Mode", window_name, &contour_cfg.mode, contour_cfg.MAX_MODE, updateImage);

  createTrackbar("Save Image", window_name, &SAVE_IMAGE, 1, updateImage);

  updateImage(0, 0);  // Call the function to initialize

  /// Wait until the user finishes the program
  waitKey();
  return 0;
}
