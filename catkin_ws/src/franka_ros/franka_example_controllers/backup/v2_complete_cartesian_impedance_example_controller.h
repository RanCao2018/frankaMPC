// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <mutex>

#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <franka_msgs/ImpedanceControllerVariable.h>
#include <franka_msgs/ErrorState.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_example_controllers/complete_compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/trigger_rate.h>

namespace franka_example_controllers {

class CompleteCartesianImpedanceExampleController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double filter_params_{0.005};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{20.0};
  const double delta_tau_max_{1.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Matrix<double, 6, 1> error_state_;
  Eigen::Matrix<double, 6, 7> last_jacobian_;
  Eigen::Matrix<double, 6, 1> u_mpc_;
  Eigen::Matrix<double, 6, 1> u_mpc_target_;
  Eigen::Vector3d position_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Vector3d velocity_d_;
  Eigen::Vector3d velocity_;
  Eigen::Vector3d accel_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Vector3d velocity_d_target_;
  Eigen::Vector3d accel_d_target_;
  Eigen::Quaterniond orientation_d_target_;

///< Publisher for the error state.
  realtime_tools::RealtimePublisher<franka_msgs::ErrorState> error_state_pub_;
  ///< Rate to trigger publishing the current error state.
  franka_hw::TriggerRate publish_rate_;

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::complete_compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::complete_compliance_paramConfig& config,
                               uint32_t level);

  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;
  ros::Subscriber sub_mpc_command_;
  void equilibriumPoseCallback(const franka_msgs::ImpedanceControllerVariableConstPtr& msg);
  void mpcCommandCallback(const geometry_msgs::WrenchConstPtr& msg);
  /**
   * Publishes a error state for MPC.
   */
  void publishErrorState();
};

}  // namespace franka_example_controllers
