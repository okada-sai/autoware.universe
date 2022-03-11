// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Author: v1.0 Yukihiro Saito
//

#ifndef MULTI_OBJECT_TRACKER__TRACKER__MODEL__NORMAL_VEHICLE_TRACKER_HPP_
#define MULTI_OBJECT_TRACKER__TRACKER__MODEL__NORMAL_VEHICLE_TRACKER_HPP_

#include "tracker_base.hpp"

#include <kalman_filter/kalman_filter.hpp>

class NormalVehicleTracker : public Tracker
{
private:
  autoware_auto_perception_msgs::msg::DetectedObject object_;
  rclcpp::Logger logger_;

private:
  KalmanFilter ekf_;
  rclcpp::Time last_update_time_;
  enum IDX {
    X = 0,
    Y = 1,
    YAW = 2,
    VX = 3,
    WZ = 4,
  };
  struct EkfParams
  {
    char dim_x = 5;
    double q_cov_x;
    double q_cov_y;
    double q_cov_yaw;
    double q_cov_wz;
    double q_cov_vx;
    double p0_cov_vx;
    double p0_cov_wz;
    // if use_measurement_covariance_ is false, use following params
    bool use_measurement_covariance;
    double r_cov_x;
    double r_cov_y;
    double r_cov_yaw;
    double p0_cov_x;
    double p0_cov_y;
    double p0_cov_yaw;
  } ekf_params_;

  double max_vx_;
  double max_wz_;
  double z_;

private:
  struct BoundingBox
  {
    double width;
    double length;
    double height;
  };
  BoundingBox bounding_box_;

public:
  NormalVehicleTracker(
    const rclcpp::Time & time, const autoware_auto_perception_msgs::msg::DetectedObject & object);

  bool predict(const rclcpp::Time & time) override;
  bool predict(const double dt, KalmanFilter & ekf) const;
  bool measure(
    const autoware_auto_perception_msgs::msg::DetectedObject & object,
    const rclcpp::Time & time) override;
  bool measureWithPose(const autoware_auto_perception_msgs::msg::DetectedObject & object);
  bool measureWithShape(const autoware_auto_perception_msgs::msg::DetectedObject & object);
  bool getTrackedObject(
    const rclcpp::Time & time,
    autoware_auto_perception_msgs::msg::TrackedObject & object) const override;
  virtual ~NormalVehicleTracker() {}
};

#endif  // MULTI_OBJECT_TRACKER__TRACKER__MODEL__NORMAL_VEHICLE_TRACKER_HPP_
