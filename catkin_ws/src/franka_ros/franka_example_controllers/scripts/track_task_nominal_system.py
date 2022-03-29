#!/usr/bin/env python

from cmath import phase
import rospy
#import rosbag
import tf.transformations
import numpy as np
import copy

#from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
from franka_msgs.msg import ImpedanceControllerVariable

# bag = rosbag.Bag('/home/fr/caoRan/frankaMPC/test.bag', 'w')
marker_pose = ImpedanceControllerVariable()
pose_pub = None
sampleTime = 0.001
trajectory_period = 30.
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]


def wait_for_initial_pose():
    msg = rospy.wait_for_message("franka_state_controller/franka_states",
                                 FrankaState)  # type: FrankaState

    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(msg.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / \
        np.linalg.norm(initial_quaternion)
    marker_pose.pose_d.pose.orientation.x = initial_quaternion[0]
    marker_pose.pose_d.pose.orientation.y = initial_quaternion[1]
    marker_pose.pose_d.pose.orientation.z = initial_quaternion[2]
    marker_pose.pose_d.pose.orientation.w = initial_quaternion[3]
    marker_pose.pose_d.pose.position.x = msg.O_T_EE[12]
    marker_pose.pose_d.pose.position.y = msg.O_T_EE[13]
    marker_pose.pose_d.pose.position.z = msg.O_T_EE[14]

def circle_state_ref(iter, startPos, phase = 0.0):
    r = 0.15

    currentTime = iter * sampleTime + phase
    posY = r * np.sin(2.0 * np.pi / trajectory_period * currentTime + 3.0 * np.pi / 2.0) + r
    posZ = r * np.cos(2.0 * np.pi / trajectory_period * currentTime  + 3.0 * np.pi / 2.0)
    velY = 2.0 * np.pi / trajectory_period * r * np.cos(2.0 * np.pi / trajectory_period * currentTime + 3.0 * np.pi / 2.0)
    velZ = - 2.0 * np.pi / trajectory_period * r * np.sin(2.0 * np.pi / trajectory_period * currentTime + 3.0 * np.pi / 2.0)
    AccY = -(2.0 * np.pi / trajectory_period)**2 * r * np.sin(2.0 * np.pi / trajectory_period * currentTime + 3.0 * np.pi / 2.0)
    AccZ = -(2.0 * np.pi / trajectory_period)**2 * r * np.cos(2.0 * np.pi / trajectory_period * currentTime + 3.0 * np.pi / 2.0)

    posRef = startPos + np.array([0.0, posY, posZ])
    velRef = np.array([0.0, velY, velZ])
    AccRef = np.array([0.0, AccY, AccZ])   

    stateRef = np.concatenate((posRef, velRef, AccRef))

    return stateRef

def nominal_system_ref(iter, startPos):
    # initialization
    state = np.zeros((7, 1))
    dstate = np.zeros((7, 1))
    state[0:3] = startPos

def talker():
    rospy.init_node("equilibrium_pose_node")
    link_name = rospy.get_param("~link_name")
    pose_pub = rospy.Publisher(
        "equilibrium_pose", ImpedanceControllerVariable, queue_size=1)
    rate = rospy.Rate(1./sampleTime) 
    iter = 0

    wait_for_initial_pose()
    start_pose = copy.deepcopy(marker_pose)
    start_pose_position = np.array((start_pose.pose_d.pose.position.x, 
                    start_pose.pose_d.pose.position.y, start_pose.pose_d.pose.position.z))    

    while (not rospy.is_shutdown() and (iter <= trajectory_period/sampleTime - 1)):
        stateRef = circle_state_ref(iter, start_pose_position)
        marker_pose.header.frame_id = link_name
        marker_pose.header.stamp = rospy.Time.now()
        marker_pose.pose_d.pose.position.x = max([min([stateRef[0],
                                          position_limits[0][1]]),
                                          position_limits[0][0]])
        marker_pose.pose_d.pose.position.y = max([min([stateRef[1],
                                          position_limits[1][1]]),
                                          position_limits[1][0]])
        marker_pose.pose_d.pose.position.z = max([min([stateRef[2],
                                          position_limits[2][1]]),
                                          position_limits[2][0]])
        marker_pose.pose_d.pose.orientation = start_pose.pose_d.pose.orientation
        marker_pose.twist_d.twist.linear.x = stateRef[3]
        marker_pose.twist_d.twist.linear.y = stateRef[4]
        marker_pose.twist_d.twist.linear.z = stateRef[5]
        marker_pose.accel_d.accel.linear.x = stateRef[6]
        marker_pose.accel_d.accel.linear.y = stateRef[7]
        marker_pose.accel_d.accel.linear.z = stateRef[8]
        if (iter * sampleTime > 10.0) and (iter * sampleTime < 13.0):
            marker_pose.wrench_sim.wrench.force.z = 0.0
        else:
            marker_pose.wrench_sim.wrench.force.z = 0.0
        for i in range(15): # state_size(6), horizon(15)
            temp_stateRef = circle_state_ref(iter, start_pose_position, phase = (i+1)*1.0/100.0)
            for j in range(6):
                marker_pose.predictReference[6*i+j] = temp_stateRef[j]
        #rospy.loginfo(marker_pose)
        #bag.write('postion',)
        pose_pub.publish(marker_pose)
        iter = iter + 1
        rate.sleep()



if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        #bag.close()
        pass