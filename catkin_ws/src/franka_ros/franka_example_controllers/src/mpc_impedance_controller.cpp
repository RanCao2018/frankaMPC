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
  // sub_equilibrium_pose_ = node_handle.subscribe(
  //     "equilibrium_pose", 1, &MpcImpedanceController::equilibriumPoseCallback, this,
  //     ros::TransportHints().reliable().tcpNoDelay());
    
    cartesian_mass_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    cartesian_damping_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    cartesian_stiffness_ = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);

    matrix_a_ = Eigen::MatrixXd::Zero(generalized_state_size_, generalized_state_size_);
    matrix_ad_ = Eigen::MatrixXd::Zero(generalized_state_size_, generalized_state_size_);
    matrix_b_ = Eigen::MatrixXd::Zero(generalized_state_size_, control_size_);
    matrix_bd_ = Eigen::MatrixXd::Zero(generalized_state_size_, control_size_);
    
    x_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    state_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    state_ref_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    equilibrium_state_ = Eigen::MatrixXd::Zero(2 * basic_state_size_, horizon_+1);;
    u_ = Eigen::MatrixXd::Zero(control_size_, 1);
    z_ = 0.0;
    error_state_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);

    uMax_ = Eigen::MatrixXd::Zero(control_size_, 1);
    uMin_ = Eigen::MatrixXd::Zero(control_size_, 1);
    xMax_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);
    xMin_ = Eigen::MatrixXd::Zero(generalized_state_size_, 1);;

    Q_ = Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    R_ = Eigen::MatrixXd::Identity(control_size_, control_size_);


    // Setup publisher for the centering frame.
    publish_rate_ = franka_hw::TriggerRate(update_rate_);
    mpc_command_pub_.init(node_handle, "mpc_command", 1, true);

    sample_time_ = 1.0/update_rate_;

    return true;
}

void MpcImpedanceController::start() {
    double stiffness_parameter = 100.0;
    cartesian_stiffness_.diagonal() << stiffness_parameter, stiffness_parameter, stiffness_parameter;
    cartesian_damping_.diagonal() << 0.8*sqrt(2.0*stiffness_parameter), 0.8*sqrt(2.0*stiffness_parameter), 0.8*sqrt(2.0*stiffness_parameter);
    matrix_a_.block(0, basic_state_size_, basic_state_size_, basic_state_size_) = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    matrix_a_.block(basic_state_size_, 0, basic_state_size_, basic_state_size_) = - cartesian_mass_.inverse() * cartesian_stiffness_;
    matrix_a_.block(basic_state_size_, basic_state_size_, basic_state_size_, basic_state_size_) = - cartesian_mass_.inverse() * cartesian_damping_;
    matrix_b_.block(basic_state_size_, 0, basic_state_size_, basic_state_size_) = cartesian_mass_.inverse();
    // discrelization 
    Eigen::MatrixXd matrix_identity =  Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
    // matrix_ad_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
    //             (matrix_identity + 0.5 * matrix_a_ * sample_time_);
    // matrix_bd_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
    //              matrix_b_ * sample_time_;
    matrix_ad_ = matrix_identity + matrix_a_ * sample_time_;
    matrix_bd_ = matrix_b_ * sample_time_;

    // input inequality constraints
    uMin_ << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY;

    uMax_ << OsqpEigen::INFTY,OsqpEigen::INFTY,OsqpEigen::INFTY;

    // state inequality constraints
    xMin_ << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY,
        -OsqpEigen::INFTY, -OsqpEigen::INFTY,-OsqpEigen::INFTY, -1e-3;

    xMax_ << OsqpEigen::INFTY,OsqpEigen::INFTY,OsqpEigen::INFTY,
        OsqpEigen::INFTY, OsqpEigen::INFTY,OsqpEigen::INFTY, OsqpEigen::INFTY;

    // wieght matrices
    Q_.block(0, 0, basic_state_size_, basic_state_size_) = 1000.0 * Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    Q_.block(basic_state_size_, basic_state_size_, basic_state_size_, basic_state_size_) = Eigen::MatrixXd::Identity(basic_state_size_, basic_state_size_);
    Q_(generalized_state_size_-1, generalized_state_size_-1) = 0.0;
    R_ = Eigen::MatrixXd::Identity(control_size_, control_size_);

    // cast the MPC problem as QP problem
    castMPCToQPHessian();
    castMPCToQPGradient();
    castMPCToQPConstraintMatrix();
    castMPCToQPConstraintVectors();

    // settings
    //solver_.settings()->setVerbosity(false);
    //solver_.settings()->setWarmStart(true);
    solver_.settings()->setPolish(true);
    //solver_.settings()->setScaling(0);
    //solver_.settings()->setScaledTerimination(false);
    // solver_.settings()->setAbsoluteTolerance(5e-5);
    // solver_.settings()->setRelativeTolerance(5e-5);
    solver_.settings()->setPrimalInfeasibilityTollerance(1e-5);

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
  lowerBound_.block(0, 0, generalized_state_size_, 1) = -x_;
  upperBound_.block(0, 0, generalized_state_size_, 1) = -x_;

  for (size_t i = 0; i < horizon_; i++) {
    lowerBound_(generalized_state_size_ * (i+1) + 6, 0) = z_e_;
    upperBound_(generalized_state_size_ * (i+1) + 6, 0) = z_e_;
  }
  Eigen::VectorXd state_limit(2*basic_state_size_);
  state_limit << OsqpEigen::INFTY, 0.28, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY;

  for (size_t i = 0; i < horizon_+1; i++) {
    xMax_.head(2*basic_state_size_) << state_limit - equilibrium_state_.col(i);
    upperBound_.block(generalized_state_size_ * (horizon_ + 1) + generalized_state_size_ * i, 0, generalized_state_size_, 1) = xMax_;
  } // horizon_+1 means terminal constraint
}

void MpcImpedanceController::updateModelMatrix() {
  matrix_a_.block(generalized_state_size_-1, 0, 1, basic_state_size_) = -x_.head(basic_state_size_).transpose() * 0.0;
  matrix_a_.block(generalized_state_size_-1, basic_state_size_, 1, basic_state_size_) = 2.0 * x_.segment(basic_state_size_, basic_state_size_).transpose() * cartesian_damping_ - u_.transpose(); 
  matrix_b_.block(generalized_state_size_-1, 0, 1, basic_state_size_) = -x_.segment(basic_state_size_, basic_state_size_).transpose();

  Eigen::MatrixXd matrix_identity =  Eigen::MatrixXd::Identity(generalized_state_size_, generalized_state_size_);
  // matrix_ad_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
  //             (matrix_identity + 0.5 * matrix_a_ * sample_time_);
  // matrix_bd_ = (matrix_identity - 0.5 * matrix_a_ * sample_time_).inverse() * 
  //               matrix_b_ * sample_time_;
  matrix_ad_ = matrix_identity + matrix_a_ * sample_time_;
  matrix_bd_ = matrix_b_ * sample_time_;

  //castMPCToQPConstraintMatrix();
  // update model matrix
  for (int i = 0; i < horizon_; i++) {
    for (int j = 0; j < generalized_state_size_; j++) {
      for (int k = 0; k < generalized_state_size_; k++) {
        float value = matrix_ad_(j, k);
        if (value != 0) {
          linearMatrix_.coeffRef(generalized_state_size_ * (i + 1) + j, generalized_state_size_ * i + k) = value;
        }
      }
      for (int k = 0; k < control_size_; k++) {
        float value = matrix_bd_(j, k);
        if (value != 0) {
          linearMatrix_.coeffRef(generalized_state_size_ * (i + 1) + j, control_size_ * i + k + generalized_state_size_ * (horizon_ + 1)) = value;
        }
      }
    }
  }
}
void MpcImpedanceController::update() {
  //ros::Time begin1 = ros::Time::now();
  
  x_ = state_ - state_ref_;
  object_ = (x_.transpose() * Q_ * x_ + u_.transpose() * R_ * u_).value();
  ax = (2.0 * x_.segment(basic_state_size_, basic_state_size_).transpose() * cartesian_damping_ - u_.transpose()) * x_.segment(basic_state_size_, basic_state_size_); 
  bu = -x_.segment(basic_state_size_, basic_state_size_).transpose() * u_; 
                                         
  // update the constraint bound (Note: the initial value is also contained in constraints)
  updateConstraintVectors();
  if(!solver_.updateBounds(lowerBound_, upperBound_)) ROS_ERROR("cannot update bound!");
  //ros::Time begin2 = ros::Time::now();
  updateModelMatrix();
  if(!solver_.updateLinearConstraintsMatrix(linearMatrix_)) ROS_ERROR("updateLinearConstraintsMatrix error");
  //ros::Time begin3 = ros::Time::now();
  // solve the QP problem
  if(!solver_.solve()) 
    ROS_ERROR("cannot solve!");
  // get the controller input
  QPSolution_ = solver_.getSolution();
  u_ = QPSolution_.block(generalized_state_size_ * (horizon_ + 1), 0, control_size_, 1); 
  z_ = QPSolution_(generalized_state_size_*2-1, 0);
  z_e_ = QPSolution_(generalized_state_size_*3-1, 0) - QPSolution_(generalized_state_size_*2-1, 0);
  
  //dot_z_c = (x_.segment(basic_state_size_, basic_state_size_).transpose() * cartesian_damping_ + u_.transpose()) * x_.segment(basic_state_size_, basic_state_size_) * sample_time_;
 
  // if (publish_rate_()) {
  //   publishMPCCommand();
  // }
  //ros::Time end = ros::Time::now();
  //ROS_INFO("-------------------constraint: %fs, model: %fs, solve: %fs-------------------", (begin2 - begin1).toSec(), (begin3 - begin2).toSec(), (end - begin3).toSec());
  ROS_INFO("-------------------state: %f, %f, %f-------------------", object_, ax.value(), bu.value());
  //ROS_INFO("-------------------state_ref: %f, %f, %f, %f-------------------", state_ref_(1), state_ref_(2), state_ref_(4), state_ref_(5));
  // Eigen::VectorXd opti_value_mpc = (x_).transpose() * Q_ * (x_) + u_.transpose() * R_ * u_;
  // simModelUpdate();
}

void MpcImpedanceController::simModelUpdate() {
  z_ = (matrix_ad_ * x_ + matrix_bd_ * u_)(generalized_state_size_-1);
}

void MpcImpedanceController::errorStateCallback(const franka_msgs::ErrorStateConstPtr& msg) {
    std::lock_guard<std::mutex> error_state_mutex_lock(
      error_state_mutex_);
    for (size_t i = 0; i < basic_state_size_; i++) {
        state_[i] = msg->pose[i];
        state_[i+3] = msg->velocity[i];
    }
    for (size_t i = 0; i < 2*basic_state_size_; i++) {
         error_state_[i] = msg->error_state[i];
    }
    state_[generalized_state_size_-1] = z_;
    for (size_t i = 0; i < horizon_ + 1; i++) {
      for (size_t j = 0; j < 2*basic_state_size_; j++) {
        equilibrium_state_(j,i) = msg->predict_reference[2*basic_state_size_*i+j];
      }
    }
    state_ref_.head(2*basic_state_size_) = equilibrium_state_.col(0);
 }

// void MpcImpedanceController::equilibriumPoseCallback(const franka_msgs::ImpedanceControllerVariableConstPtr& msg) {
//     std::lock_guard<std::mutex> equilibrium_mutex_lock(
//       equilibrium_mutex_);
//     equilibrium_state_.col(0).head(basic_state_size_) << msg->pose_d.pose.position.x, msg->pose_d.pose.position.y, msg->pose_d.pose.position.z;
//     equilibrium_state_.col(0).tail(basic_state_size_) << msg->twist_d.twist.linear.x, msg->twist_d.twist.linear.y, msg->twist_d.twist.linear.z;
//     for (size_t i = 0; i < horizon_; i++) {
//       for (size_t j = 0; j < 2*basic_state_size_; j++) {
//         equilibrium_state_(j,i+1) = msg->predictReference[2*basic_state_size_*i+j];
//       }
//     }
//     state_ref_.head(2*basic_state_size_) = equilibrium_state_.col(0);
//  }
void MpcImpedanceController::publishMPCCommand() {
    if (mpc_command_pub_.trylock()) {
      mpc_command_pub_.msg_.command.force.x = u_[0];
      mpc_command_pub_.msg_.command.force.y = u_[1];
      mpc_command_pub_.msg_.command.force.z = u_[2];
      mpc_command_pub_.msg_.z = z_;
      for (size_t i = 0; i < 6; i++) {
         mpc_command_pub_.msg_.debug[i] = x_[i] - error_state_[i];
      }
    }

    mpc_command_pub_.unlockAndPublish();
    //ROS_INFO("-------------------umpc: %f, %f, %f-------------------", u_[0], u_[1], u_[2]);
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