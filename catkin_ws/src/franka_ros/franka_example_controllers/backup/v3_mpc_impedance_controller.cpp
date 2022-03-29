#include <franka_example_controllers/mpc_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h>

#include <franka_example_controllers/pseudo_inversion.h>

namespace franka_example_controllers {

bool MpcImpedanceController::init(ros::NodeHandle& node_handle) {
  sub_error_state_ = node_handle.subscribe(
      "error_state", 1, &MpcImpedanceController::errorStateCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 1, &MpcImpedanceController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  // std::string arm_id;
  // if (!node_handle.getParam("arm_id", arm_id)) {
  //   ROS_ERROR_STREAM("MpcImpedanceController: Could not read parameter arm_id");
  //   return false;
  // }
  // std::vector<std::string> joint_names;
  // if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
  //   ROS_ERROR(
  //       "MpcImpedanceController: Invalid or no joint_names parameters provided, "
  //       "aborting controller init!");
  //   return false;
  // }

    // dynamic_reconfigure_compliance_param_node_ =
    //   ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_compliance_param_node");

    // dynamic_server_compliance_param_ = std::make_unique<
    //   dynamic_reconfigure::Server<franka_example_controllers::complete_compliance_paramConfig>>(
    //   dynamic_reconfigure_compliance_param_node_);
    // dynamic_server_compliance_param_->setCallback(
    //   boost::bind(&MpcImpedanceController::complianceParamCallback, this, _1, _2));
    
    cartesian_mass_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    cartesian_damping_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    cartesian_stiffness_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);

    matrix_a_ = Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    matrix_ad_ = Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    matrix_b_ = Eigen::MatrixXd::Zero(generalized_state_size_, control_size_);
    matrix_bd_ = Eigen::MatrixXd::Zero(generalized_state_size_, control_size_);
    
    x_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    state_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    state_ref_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    equilibrium_state_ = Eigen::MatrixXd::Zero(2 * basic_state_size_, horizon_+1);;
    u_ = Eigen::MatrixXd::Zero(control_size_, 1);

    uMax_ = Eigen::MatrixXd::Zero(control_size_, 1);
    uMin_ = Eigen::MatrixXd::Zero(control_size_, 1);
    xMax_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    xMin_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);;

    Q_ = Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    R_ = Eigen::MatrixXd::Identity(control_size_, control_size_);


    // Setup publisher for the centering frame.
    publish_rate_ = franka_hw::TriggerRate(100.0);
    mpc_command_pub_.init(node_handle, "mpc_command", 1, true);

    return true;
}

void MpcImpedanceController::start() {
    double stiffness_parameter = 100.0;
    cartesian_stiffness_.diagonal() << stiffness_parameter, stiffness_parameter, stiffness_parameter;
    cartesian_damping_.diagonal() << 2.0*sqrt(stiffness_parameter), 2.0*sqrt(stiffness_parameter), 2.0*sqrt(stiffness_parameter);
    matrix_a_.block(0, 0, basic_state_size_, basic_state_size_) = Eigen::MatrixXd::Zero(basic_state_size_, basic_state_size_);
    matrix_a_.block(0, basic_state_size_, basic_state_size_, basic_state_size_) = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    matrix_a_.block(basic_state_size_, 0, basic_state_size_, basic_state_size_) = - cartesian_mass_.inverse() * cartesian_stiffness_;
    matrix_a_.block(basic_state_size_, basic_state_size_, basic_state_size_, basic_state_size_) = - cartesian_mass_.inverse() * cartesian_damping_;


    matrix_b_.block(basic_state_size_, 0, basic_state_size_, basic_state_size_) = - cartesian_mass_.inverse();
    // discrelization 
    Eigen::MatrixXd matrix_identity =  Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    matrix_ad_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
                (matrix_identity + 0.5 * matrix_a_ * sample_time_);
    matrix_bd_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
                 matrix_b_ * sample_time_;

    // input inequality constraints
    uMin_ << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY;

    uMax_ << OsqpEigen::INFTY,OsqpEigen::INFTY,OsqpEigen::INFTY;

    // state inequality constraints
    xMin_ << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY,
        -OsqpEigen::INFTY, -OsqpEigen::INFTY,-OsqpEigen::INFTY;

    xMax_ << OsqpEigen::INFTY,OsqpEigen::INFTY,OsqpEigen::INFTY,
        OsqpEigen::INFTY, OsqpEigen::INFTY,OsqpEigen::INFTY;

    // wieght matrices
    Q_ = 100.0 * Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    R_ = Eigen::MatrixXd::Identity(control_size_, control_size_);

    // cast the MPC problem as QP problem
    castMPCToQPHessian();
    castMPCToQPGradient();
    castMPCToQPConstraintMatrix();
    castMPCToQPConstraintVectors();

    // settings
    //solver.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setPolish(true);
    solver_.settings()->setScaling(0);
    solver_.settings()->setScaledTerimination(true);

    // set the initial data of the QP solver
    solver_.data()->setNumberOfVariables(generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_);
    solver_.data()->setNumberOfConstraints(2 * generalized_state_size_ * (horizon_ + 1) +  control_size_ * horizon_);
    if(!solver_.data()->setHessianMatrix(hessianMatrix_))  ROS_ERROR("setHessianMatrix error");
    if(!solver_.data()->setGradient(gradient_)) ROS_ERROR("setGradient error");
    if(!solver_.data()->setLinearConstraintsMatrix(linearMatrix_)) ROS_ERROR("setLinearConstraintsMatrix error");
    if(!solver_.data()->setLowerBound(lowerBound_)) ROS_ERROR("setLowerBound error");
    if(!solver_.data()->setUpperBound(upperBound_))ROS_ERROR("setUpperBound error");

    // instantiate the solver
    if(!solver_.initSolver()) ROS_ERROR("initSolver error");
}

void MpcImpedanceController::castMPCToQPHessian() {
  hessianMatrix_.resize(generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_, generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_);

  // populate hessian matrix
  for (int i = 0; i < generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_; i++) {
    if (i < generalized_state_size_ * (horizon_ + 1)) {
      int posQ = i % generalized_state_size_;
      float value = Q_.diagonal()[posQ];
      if (value != 0)
        hessianMatrix_.insert(i, i) = value;
    } else {
      int posR = i % control_size_;
      float value = R_.diagonal()[posR];
      if (value != 0)
        hessianMatrix_.insert(i, i) = value;
    }
  }
}

void MpcImpedanceController::castMPCToQPGradient() {
  Eigen::MatrixXd Qx_ref;
  Qx_ref = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
  Qx_ref = -1.0 * Q_ * Eigen::MatrixXd::Zero(generalized_state_size_, 1);

  // populate the gradient vector
  gradient_ = Eigen::VectorXd::Zero(generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_, 1);
  for (int i = 0; i < generalized_state_size_ * (horizon_ + 1); i++) {
    int posQ = i % generalized_state_size_;
    float value = Qx_ref(posQ, 0);
    gradient_(i, 0) = value;
  }
}

void MpcImpedanceController::castMPCToQPConstraintVectors() {
  // evaluate the lower and the upper inequality vectors
  Eigen::VectorXd lowerInequality = Eigen::VectorXd::Zero(
    generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_);  
  Eigen::VectorXd upperInequality = Eigen::VectorXd::Zero(
    generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_);
  for (size_t i = 0; i < horizon_; i++) {
    lowerInequality.block(control_size_ * i + generalized_state_size_ * (horizon_ + 1), 0,
                          control_size_, 1) = uMin_;
    upperInequality.block(control_size_ * i + generalized_state_size_ * (horizon_ + 1), 0,
                          control_size_, 1) = uMax_;
  }
  for (size_t i = 0; i < horizon_ + 1; i++) {
    lowerInequality.block(generalized_state_size_ * i, 0, generalized_state_size_, 1) = xMin_;
    upperInequality.block(generalized_state_size_ * i, 0, generalized_state_size_, 1) = xMax_;
  }
  Eigen::VectorXd lowerEquality = Eigen::VectorXd::Zero(
    generalized_state_size_ * (horizon_ + 1));
  Eigen::VectorXd upperEquality;
  lowerEquality.block(0, 0, generalized_state_size_, 1) = -x_;
  upperEquality = lowerEquality;
  lowerEquality = lowerEquality;

  // merge inequality and equality vectors
  lowerBound_ = Eigen::VectorXd::Zero(2 * generalized_state_size_*(horizon_+1) +  control_size_ * horizon_);
  lowerBound_ << lowerEquality, lowerInequality;

  upperBound_ = Eigen::VectorXd::Zero(2 * generalized_state_size_*(horizon_+1) +  control_size_ * horizon_);
  upperBound_ << upperEquality, upperInequality;  
}

void MpcImpedanceController::castMPCToQPConstraintMatrix() {
  linearMatrix_.resize(generalized_state_size_ * (horizon_ + 1) +
                       generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_, 
                       generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_);
  // populate linear constraint matrix
  for (int i = 0; i < generalized_state_size_ * (horizon_ + 1); i++) {
    linearMatrix_.insert(i, i) = -1;
  }

  for (int i = 0; i < horizon_; i++)
    for (int j = 0; j < generalized_state_size_; j++)
      for (int k = 0; k < generalized_state_size_; k++) {
        float value = matrix_ad_(j, k);
        if (value != 0) {
          linearMatrix_.insert(generalized_state_size_ * (i + 1) + j, generalized_state_size_ * i + k) = value;
        }
      }

  for (int i = 0; i < horizon_; i++)
    for (int j = 0; j < generalized_state_size_; j++)
      for (int k = 0; k < control_size_; k++) {
        float value = matrix_bd_(j, k);
        if (value != 0) {
          linearMatrix_.insert(generalized_state_size_ * (i + 1) + j, control_size_ * i + k + generalized_state_size_ * (horizon_ + 1)) = value;
        }
      }

  for (int i = 0; i < generalized_state_size_ * (horizon_ + 1) + control_size_ * horizon_; i++) {
    linearMatrix_.insert(i + (horizon_ + 1) * generalized_state_size_, i) = 1;
  }
}

void MpcImpedanceController::updateConstraintVectors() {
  Eigen::VectorXd state_limit(generalized_state_size_);
  state_limit << OsqpEigen::INFTY, 0.18, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY;
  lowerBound_.block(0, 0, generalized_state_size_, 1) = -x_;
  upperBound_.block(0, 0, generalized_state_size_, 1) = -x_;

  for (size_t i = 0; i < horizon_+1; i++) {
    xMax_ << state_limit - equilibrium_state_.col(i);
    upperBound_.block(generalized_state_size_ * (horizon_ + 1) + generalized_state_size_ * i, 0, generalized_state_size_, 1) = xMax_;
  } // horizon_+1 means terminal constraint
}

void MpcImpedanceController::update() {

  x_ = state_ - state_ref_;                                               
  // update the constraint bound (Note: the initial value is also contained in constraints)
  updateConstraintVectors();
  if(!solver_.updateBounds(lowerBound_, upperBound_)) ROS_ERROR("cannot update bound!");
  
  // solve the QP problem
  if(!solver_.solve()) ROS_ERROR("cannot solve!");
  // get the controller input
  QPSolution_ = solver_.getSolution();
  u_ = QPSolution_.block(generalized_state_size_ * (horizon_ + 1), 0, control_size_, 1); 

  // double debug_u_norm = u_(1);

  if (publish_rate_()) {
    publishMPCCommand();
  }
  ROS_INFO("-------------------state: %f, %f, %f, %f-------------------", state_(1), state_(2), state_(4), state_(5));
  ROS_INFO("-------------------state_ref: %f, %f, %f, %f-------------------", state_ref_(1), state_ref_(2), state_ref_(4), state_ref_(5));
  // Eigen::VectorXd opti_value_mpc = (x_).transpose() * Q_ * (x_) + u_.transpose() * R_ * u_;
  // simModelUpdate();


}

void MpcImpedanceController::simModelUpdate() {
  x_ = matrix_ad_ * x_ + matrix_bd_ * u_;
}

void MpcImpedanceController::complianceParamCallback(
    franka_example_controllers::complete_compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_.setIdentity(basic_state_size_, basic_state_size_);
  cartesian_stiffness_
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_.setIdentity(basic_state_size_, basic_state_size_);
  // Damping ratio = 1
  cartesian_damping_
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
}

void MpcImpedanceController::errorStateCallback(const franka_msgs::ErrorStateConstPtr& msg) {
    std::lock_guard<std::mutex> error_state_mutex_lock(
      error_state_mutex_);
    for (size_t i = 0; i <= generalized_state_size_ - 1; i++) {
        state_[i] = msg->error_state[i];
    }
 }

void MpcImpedanceController::equilibriumPoseCallback(const franka_msgs::ImpedanceControllerVariableConstPtr& msg) {
    std::lock_guard<std::mutex> error_state_mutex_lock(
      error_state_mutex_);
    equilibrium_state_.col(0).head(basic_state_size_) << msg->pose_d.pose.position.x, msg->pose_d.pose.position.y, msg->pose_d.pose.position.z;
    equilibrium_state_.col(0).tail(basic_state_size_) << msg->twist_d.twist.linear.x, msg->twist_d.twist.linear.y, msg->twist_d.twist.linear.z;
    for (size_t i = 1; i < horizon_+1; i++) {
      for (size_t j = 0; j < 2*basic_state_size_; j++) {
        equilibrium_state_(j,i) = msg->predictReference[2*basic_state_size_*i+j];
      }
    }
    state_ref_ = equilibrium_state_.col(0);
 }
void MpcImpedanceController::publishMPCCommand() {
    if (mpc_command_pub_.trylock()) {
      mpc_command_pub_.msg_.force.x = u_[0];
      mpc_command_pub_.msg_.force.y = u_[1];
      mpc_command_pub_.msg_.force.z = u_[2];
    }
    mpc_command_pub_.unlockAndPublish();
    ROS_INFO("-------------------umpc: %f, %f, %f-------------------", u_[0], u_[1], u_[2]);
}

}  // namespace franka_example_controllers

// PLUGINLIB_EXPORT_CLASS(franka_example_controllers::MpcImpedanceController,
//                        controller_interface::ControllerBase)

// int main(int argc, char** argv) {
//   ros::init(argc, argv, "MPC_controller");
//   ros::NodeHandle node_handle("~");
//   // franka_hw::FrankaHW franka_control; 

//   double publish_rate = 1000;
//   node_handle.getParam("publish_rate", publish_rate);
//   ros:: Rate rate(publish_rate);

//   franka_example_controllers::MpcImpedanceController mpcControl;

//   mpcControl.init(node_handle);

//   const int numberOfSteps = 50;
//   int iter = 0;

//   while (ros::ok() && (iter < numberOfSteps)) {

//     mpcControl.rollout();

//     ros::spinOnce();
//     rate.sleep();
//   }

// }