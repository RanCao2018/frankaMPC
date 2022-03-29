// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/complete_cartesian_impedance_example_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace franka_example_controllers {

bool CompleteCartesianImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &CompleteCartesianImpedanceExampleController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  sub_mpc_command_ = node_handle.subscribe(
      "mpc_command", 20, &CompleteCartesianImpedanceExampleController::mpcCommandCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CompleteCartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CompleteCartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CompleteCartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CompleteCartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CompleteCartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CompleteCartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CompleteCartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CompleteCartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::complete_compliance_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CompleteCartesianImpedanceExampleController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  velocity_d_.setZero();
  velocity_d_target_.setZero();
  accel_d_.setZero();
  accel_d_target_.setZero();
  error_state_.setZero();
  last_jacobian_.setZero();
  u_mpc_.setZero();
  u_mpc_target_.setZero();

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  // Setup publisher for the centering frame.
  publish_rate_ = franka_hw::TriggerRate(100.0);
  error_state_pub_.init(node_handle, "error_state", 1, true);


  return true;
}

void CompleteCartesianImpedanceExampleController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());
  velocity_d_ = (jacobian * dq_initial).head(3);
  velocity_d_target_ = (jacobian * dq_initial).head(3);
  last_jacobian_ = jacobian;

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
}

void CompleteCartesianImpedanceExampleController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& /*period*/) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 49> inertia_array = model_handle_->getMass();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> inertia(inertia_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());

  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d_;

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.linear() * error.tail(3);

  // velocity error
  Eigen::Matrix<double, 6, 1> twist(jacobian*dq);
  Eigen::Matrix<double, 6, 1> derror;
  // flit velocity signal
  velocity_ << twist.head(3);
  // velocity << 0.01 * velocity + (1.0 - 0.01) * last_velocity_;
  // last_velocity_ << velocity;
  derror.head(3) << velocity_ - velocity_d_;
  derror.tail(3).setZero();

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv, jacobian_pinv, djacobian;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  pseudoInverse(jacobian, jacobian_pinv);
  djacobian = (jacobian - last_jacobian_)/0.001;
  last_jacobian_ = jacobian;

  // complete impedance controller
  Eigen::Matrix<double, 6, 1> PD, u, accel_d_la_;
  Eigen::VectorXd v(7);
  accel_d_la_.setZero();
  accel_d_la_.head(3) << accel_d_;
  PD << -cartesian_stiffness_ * error - cartesian_damping_ * derror;
  u << accel_d_la_ + PD - u_mpc_;
  // ROS_INFO("-------------------umpc: %f, %f, %f-------------------", u_mpc_[0], u_mpc_[1], u_mpc_[2]);
  v << jacobian_pinv * (u - djacobian * dq);

  // Cartesian PD control with damping ratio = 1
  tau_task << inertia * v;
  // original impedance controller
  // tau_task << jacobian.transpose() *
  //               (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

  // nullspace PD control with damping ratio = 1
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                        (2.0 * sqrt(nullspace_stiffness_)) * dq);
  // Desired torque
  tau_d << tau_task + tau_nullspace + coriolis;
  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }

  error_state_.head(3) << error.head(3);
  error_state_.tail(3) << derror.head(3);
  if (publish_rate_()) {
    publishErrorState();
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  // std::cout << "car_stiff = " << cartesian_stiffness_(0,0) << " car_damp = " << cartesian_damping_(0,0) << " null_stiff = " << nullspace_stiffness_ << std::endl;
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  u_mpc_ = u_mpc_target_;
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  velocity_d_ = filter_params_ * velocity_d_target_ + (1.0 - filter_params_) * velocity_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  accel_d_ = filter_params_ * accel_d_target_ + (1.0 - filter_params_) * accel_d_;
}

Eigen::Matrix<double, 7, 1> CompleteCartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void CompleteCartesianImpedanceExampleController::complianceParamCallback(
    franka_example_controllers::complete_compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;
}

void CompleteCartesianImpedanceExampleController::equilibriumPoseCallback(
    const franka_msgs::ImpedanceControllerVariableConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose_d.pose.position.x, msg->pose_d.pose.position.y, msg->pose_d.pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose_d.pose.orientation.x, msg->pose_d.pose.orientation.y,
      msg->pose_d.pose.orientation.z, msg->pose_d.pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
  velocity_d_target_ << msg->twist_d.twist.linear.x, msg->twist_d.twist.linear.y, msg->twist_d.twist.linear.z;
  accel_d_target_ << msg->accel_d.accel.linear.x, msg->accel_d.accel.linear.y, msg->accel_d.accel.linear.z;
}

void CompleteCartesianImpedanceExampleController::mpcCommandCallback(
    const geometry_msgs::WrenchConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  u_mpc_target_.head(3) << msg->force.x, msg->force.y, msg->force.z;
}

void CompleteCartesianImpedanceExampleController::publishErrorState() {
  if (error_state_pub_.trylock()) {
    for(size_t i = 0; i < 6; i++){
    error_state_pub_.msg_.error_state[i] = error_state_[i];
    error_state_pub_.msg_.equilibrium_state[i] = i < 3 ? position_d_target_[i] : velocity_d_target_[i-3];
    if (i < 3) error_state_pub_.msg_.velocity[i] = velocity_[i];
    }
    error_state_pub_.unlockAndPublish();
  }
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CompleteCartesianImpedanceExampleController,
                       controller_interface::ControllerBase)
