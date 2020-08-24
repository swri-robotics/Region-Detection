#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <boost/filesystem.hpp>

#include <iostream>

using namespace cv;

/** Global Variables */
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static cv::Mat im_frame;

static void updateImage()
{
  Mat inverted, frame_HSV, frame_threshold;

  // inverting
  // inverted = ~im_frame;
  inverted = im_frame;

  // Convert from BGR to HSV colorspace
  cvtColor(inverted, frame_HSV, COLOR_BGR2HSV);

  decltype(frame_HSV) dilation_dst = frame_HSV;

  /*
    int dilation_type = 0;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( dilation_type,
                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                         Point( dilation_size, dilation_size ) );
    dilate( frame_HSV, dilation_dst, element );
  */

  // Detect the object based on HSV Range Values
  inRange(dilation_dst, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
  //! [while]

  //! [show]
  // Show the frames
  // imshow(window_capture_name, frame);
  imshow(window_detection_name, frame_threshold);
}

//! [low]
static void on_low_H_thresh_trackbar(int, void*)
{
  low_H = min(high_H - 1, low_H);
  setTrackbarPos("Low H", window_detection_name, low_H);
  updateImage();
}
//! [low]

//! [high]
static void on_high_H_thresh_trackbar(int, void*)
{
  high_H = max(high_H, low_H + 1);
  setTrackbarPos("High H", window_detection_name, high_H);
  updateImage();
}

//! [high]
static void on_low_S_thresh_trackbar(int, void*)
{
  low_S = min(high_S - 1, low_S);
  setTrackbarPos("Low S", window_detection_name, low_S);
  updateImage();
}

static void on_high_S_thresh_trackbar(int, void*)
{
  high_S = max(high_S, low_S + 1);
  setTrackbarPos("High S", window_detection_name, high_S);
  updateImage();
}

static void on_low_V_thresh_trackbar(int, void*)
{
  low_V = min(high_V - 1, low_V);
  setTrackbarPos("Low V", window_detection_name, low_V);
  updateImage();
}

static void on_high_V_thresh_trackbar(int, void*)
{
  high_V = max(high_V, low_V + 1);
  setTrackbarPos("High V", window_detection_name, high_V);
  updateImage();
}

int main(int argc, char* argv[])
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

  im_frame = imread(image_name, IMREAD_COLOR);  // Load an image

  if (im_frame.empty())
  {
    std::cout << "Cannot read the image: " << image_name << std::endl;
    return -1;
  }

  //! [window]
  // namedWindow(window_capture_name);
  namedWindow(window_detection_name);
  //! [window]

  //! [trackbar]
  // Trackbars to set thresholds for HSV values
  createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
  createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
  createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
  createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
  createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
  createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

  createTrackbar(
      "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", window_detection_name, &dilation_elem, max_elem, [](int, void*) {
        updateImage();
      });

  createTrackbar("Kernel size:\n 2n +1", window_detection_name, &dilation_size, max_kernel_size, [](int, void*) {
    updateImage();
  });
  //! [trackbar]

  /*
    Mat frame, frame_HSV, frame_threshold;

    // Convert from BGR to HSV colorspace
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    // Detect the object based on HSV Range Values
    inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
    //! [while]

    //! [show]
    // Show the frames
    //imshow(window_capture_name, frame);
    imshow(window_detection_name, frame_threshold);
  */
  updateImage();

  waitKey();
  return 0;
}
