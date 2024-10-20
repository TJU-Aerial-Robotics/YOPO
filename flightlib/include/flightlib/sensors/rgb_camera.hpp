#pragma once

#include <yaml-cpp/yaml.h>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

enum CameraLayer { RGB = 0, DepthMap = 1, Segmentation = 2, OpticalFlow = 3 };

namespace RGBCameraTypes {
  typedef int8_t Intensity_t;
  typedef cv::Mat Image_t;

  struct RGBImage_t {
    Image_t image;
    USecs elapsed_useconds;
  };
  struct Depthmap_t {
    Image_t image;
    USecs elapsed_useconds;
  };
  struct Segement_t {
    Image_t image;
    USecs elapsed_useconds;
  };

  struct OpticFlow_t {
    Image_t image;
    USecs elapsed_useconds;
  };

  typedef Eigen::Matrix4d Mat4_t;
  typedef Eigen::Vector3d Vec3_t;

  typedef std::function<Eigen::Vector3d()> GetPos_t;
  typedef std::function<Eigen::Vector3d()> GetVel_t;
  typedef std::function<Eigen::Vector3d()> GetAcc_t;
  typedef std::function<Eigen::Quaterniond()> GetQuat_t;
  typedef std::function<Eigen::Vector3d()> GetOmega_t;
  typedef std::function<Eigen::Vector3d()> GetPsi_t;

  const std::string RGB = "rgb";
  // image post processing
  typedef std::string PostProcessingID;
  const PostProcessingID Depth = "depth";
  const PostProcessingID OpticalFlow = "optical_flow";
  const PostProcessingID ObjectSegment = "object_segment";      // object segmentation
  const PostProcessingID CategorySegment = "category_segment";  // category segmentation
}  // namespace RGBCameraTypes

class RGBCamera : SensorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RGBCamera();
  ~RGBCamera();

  // public set functions
  bool setRelPose(const Ref<Vector<3>> B_r_BC, const Ref<Matrix<3, 3>> R_BC);
  bool setWidth(const int width);
  bool setHeight(const int height);
  bool setFOV(const Scalar fov);
  bool setDepthScale(const Scalar depth_scale);
  bool setPostProcesscing(const std::vector<bool>& enabled_layers);
  bool feedImageQueue(const int image_layer, const cv::Mat& image_mat);
  void clearImageQueue();

  // public get functions
  std::vector<bool> getEnabledLayers(void) const;
  std::vector<std::string> GetPostProcessing(void);
  Matrix<4, 4> getRelPose(void) const;
  int getChannels(void) const;
  int getWidth(void) const;
  int getHeight(void) const;
  Scalar getFOV(void) const;
  Scalar getDepthScale(void) const;
  bool getRGBImage(cv::Mat& rgb_img);
  bool getDepthMap(cv::Mat& depth_map);
  bool getSegmentation(cv::Mat& segmentation);
  bool getOpticalFlow(cv::Mat& opticalflow);

  // auxiliary functions
  void enableDepth(const bool on);
  void enableOpticalFlow(const bool on);
  void enableSegmentation(const bool on);

 private:
  Logger logger_{"RBGCamera"};

  // camera parameters
  int channels_;
  int width_;
  int height_;
  Scalar fov_;
  Scalar depth_scale_;

  // Camera relative
  Vector<3> B_r_BC_;
  Matrix<4, 4> T_BC_;

  // image data buffer
  std::mutex queue_mutex_;
  const int queue_size_ = 0;  // 1

  // TODO：不要用队列，就单纯的Mat就好了，也不会有滞留的问题，也不会有弹空的问题；先不改了省的出错
  std::deque<cv::Mat> rgb_queue_;
  std::deque<cv::Mat> depth_queue_;
  std::deque<cv::Mat> opticalflow_queue_;
  std::deque<cv::Mat> segmentation_queue_;

  // [rgb, depth, segmentation, optical flow]
  std::vector<bool> enabled_layers_;
  // [depth, optical flow, segmentation, segmentation]
  std::unordered_map<RGBCameraTypes::PostProcessingID, bool> post_processing_;
};

}  // namespace flightlib
