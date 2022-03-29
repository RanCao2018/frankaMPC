#pragma once

#include <memory>
#include <mutex>

#include <string>
#include <vector>

// #include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <franka_msgs/ImpedanceControllerVariable.h>
#include <franka_msgs/MPCState.h>
#include <franka_msgs/ErrorState.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include "OsqpEigen/OsqpEigen.h"

#include <franka_example_controllers/complete_compliance_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/trigger_rate.h>

namespace franka_example_controllers{

class MpcImpedanceController{
 public:
  bool init(ros::NodeHandle& node_handle);
  void start();
  void update();
  void castMPCToQPConstraintVectors();
  void castMPCToQPConstraintMatrix();
  void castMPCToQPHessian();
  void castMPCToQPGradient();
  void updateConstraintVectors();
  void simModelUpdate();
  void updateModelMatrix();
  void stop();
  void publishMPCCommand();

 private:
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  
  // predict time interval
  const int horizon_{15};
  const double update_rate_{100.0};
  // sample_time
  double sample_time_;
  // number of generalized states
  const int generalized_state_size_ = 7;
  // number of basic states
  const int control_size_ = 3;
  const int basic_state_size_ = 3;
  // continuous and discrete system model matrices
  Eigen::MatrixXd matrix_a_;
  Eigen::MatrixXd matrix_b_;
  Eigen::MatrixXd matrix_ad_;
  Eigen::MatrixXd matrix_bd_;
  // system model parameters
  Eigen::MatrixXd cartesian_stiffness_;
  Eigen::MatrixXd cartesian_damping_;
  Eigen::MatrixXd cartesian_mass_;
  // system state matrix
  Eigen::VectorXd x_;
  Eigen::VectorXd state_;
  Eigen::VectorXd state_ref_;
  Eigen::MatrixXd equilibrium_state_;
  Eigen::VectorXd error_state_;
  double z_;
  double z_e_; //dot_z_c*t= z_c+1 - z_c
  double object_;
  Eigen::VectorXd ax;
  Eigen::VectorXd bu; 
  // system control matrix
  Eigen::VectorXd u_;
  Eigen::VectorXd command_;
  Eigen::VectorXd command_ref_;
  // objective weight
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  // system constraints including equlities and inequalities
  Eigen::VectorXd xMin_, xMax_, uMin_, uMax_;
  Eigen::VectorXd lowerBound_, upperBound_;
  // allocate QP problem matrices and vectores
  Eigen::SparseMatrix<double> hessianMatrix_;
  Eigen::VectorXd gradient_;
  Eigen::SparseMatrix<double> linearMatrix_;
  // instantiate the solver
  OsqpEigen::Solver solver_; 
  Eigen::VectorXd QPSolution_;

  std::mutex error_state_mutex_;
  std::mutex equilibrium_mutex_;

///< Publisher for the mpc control command state.
  realtime_tools::RealtimePublisher<franka_msgs::MPCState> mpc_command_pub_;
  ///< Rate to trigger publishing the current error state.
  franka_hw::TriggerRate publish_rate_;

  // Error State subscriber
  ros::Subscriber sub_error_state_;
  ros::Subscriber sub_equilibrium_pose_;
  void errorStateCallback(const franka_msgs::ErrorStateConstPtr& msg);
  void equilibriumPoseCallback(const franka_msgs::ImpedanceControllerVariableConstPtr& msg);

  /**
   * Publishes a MPC control command.
   */
  
}; 
}