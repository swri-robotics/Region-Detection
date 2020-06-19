/*
 * config_types.h
 *
 *  Created on: Jun 17, 2020
 *  Author: Jorge Nicho
 *
 * Copyright 2020 Southwest Research Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INCLUDE_CONFIG_TYPES_H_
#define INCLUDE_CONFIG_TYPES_H_

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace region_detection_core
{
  namespace config_2d
  {
    struct ThresholdCfg
    {
      bool enable = false;
      int value = 150;
      int type = cv::ThresholdTypes::THRESH_TRUNC;

      static const int MAX_VALUE = 255;
      static const int MAX_TYPE = cv::ThresholdTypes::THRESH_TOZERO_INV;
      static const int MAX_BINARY_VALUE = 255;
    };

    struct DilationCfg
    {
      bool enable = true;
      int elem = 0;
      int size = 1;

      static const int MAX_ELEM = 2;
      static const int MAX_KERNEL_SIZE = 21;
    };

    struct CannyCfg
    {
      bool enable = 1;
      int lower_threshold = 45;
      int upper_threshold = lower_threshold * 3;
      int aperture_size = 1;

      static const int MAX_ENABLE = 1;
      static const int MAX_LOWER_TH = 100;
      static const int MAX_UPPER_TH = 255;
      static const int MAX_APERTURE_SIZE = 3;
    };

    struct CountourCfg
    {
      int mode = CV_RETR_EXTERNAL;
      int method = CV_CHAIN_APPROX_SIMPLE;

      static const int MAX_MODE = CV_RETR_TREE;
      static const int MAX_METHOD = CV_CHAIN_APPROX_TC89_KCOS;
    };
  }

  namespace config_3d
  {
    struct DownsampleCfg
    {
      bool enable = true;
      double voxel_leafsize = 0.05;
    };

    struct OrderingCfg
    {
      double kdtree_leafsize = 0.05;
    };
  }
}






#endif /* INCLUDE_CONFIG_TYPES_H_ */
