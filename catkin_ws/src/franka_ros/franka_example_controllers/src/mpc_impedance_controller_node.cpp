#include <franka_example_controllers/mpc_impedance_controller.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "MPC_controller");
  ros::NodeHandle node_handle; 

  double publish_rate = 100;
  node_handle.getParam("publish_rate", publish_rate);
  ros::Rate rate(publish_rate);

  franka_example_controllers::MpcImpedanceController mpcControl;
  ROS_INFO("-------------------beign MPC-------------------");

  mpcControl.init(node_handle);
  mpcControl.start();

  // const int numberOfSteps = 50;
  int iter = 0;

  while (ros::ok()) {
    ros::Time begin = ros::Time::now();

    mpcControl.update();
    mpcControl.publishMPCCommand();

    iter = iter + 1;

    ros::spinOnce();
    rate.sleep();
    ROS_INFO("-------------------%f s-------------------", (begin - ros::Time::now()).toSec());
  }

}