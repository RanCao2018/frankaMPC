#!/usr/bin/env python
#import casadi as ca
#import casadi.tools as ca_tools

import matplotlib.pyplot as plt
import numpy as np
import time
import ctypes

import rospy
import copy
import threading
import quaternion
from franka_interface import ArmInterface
from franka_core_msgs.msg import JointCommand
from franka_msgs.msg import ErrorState
from franka_msgs.msg import FrankaState

# so_coriolis = ctypes.CDLL('src/panda_simulator/panda_simulator_examples/scripts/get_CoriolisMatrix.so')
# so_mass = ctypes.CDLL('src/panda_simulator/panda_simulator_examples/scripts/get_MassMatrix.so')
# so_friction = ctypes.CDLL('src/panda_simulator/panda_simulator_examples/scripts/get_FrictionTorque.so')
# so_gravity = ctypes.CDLL('src/panda_simulator/panda_simulator_examples/scripts/get_GravityVector.so')

# -----------------------------------------
publish_rate = 1000
# --------- Modify as required ------------
# MPC controller parameters
N = 8
sampleTime = 0.001
Q = 100* np.eye(6) # position error item weight
R = 0.1*np.eye(3) # control input item weight
R[1, 1] = 0.1

# Task-space controller parameters
# stiffness gains
P_pos = 103.41*np.eye(3)
P_pos[0,0] = 50.
P_ori = 25.
# damping gains
#D_pos = 2.*np.sqrt(2.*P_pos)
D_pos = 0.8*np.sqrt(2.*P_pos)
D_pos[0,0] = 2.*np.sqrt(2.*P_pos[0,0])
D_ori = 1.
Is_passive = False
#timeID = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
timeID = "(10)"
Is_sim = True


# -----------------------------------------

def circle_state(mpciter):
    # move along a circle on y-z plane
    r = 0.1
    period = 10
    t = mpciter * sampleTime
    pos_x = r * np.sin(2.0 * np.pi / period * t)
    pos_y = r * np.cos(2.0 * np.pi / period * t)

    vel_x = 2.0 * np.pi / period * r * np.cos(2.0 * np.pi / period * t)
    vel_y = - 2.0 * np.pi / period * r * np.sin(2.0 * np.pi / period * t)
    pos = np.array([0.0, pos_x, pos_y])
    vel = np.array([0.0, vel_x, vel_y])

    return pos, vel

def circle_state_ref(mpciter, currentTime, pos0, vel0, startPos):
    posRef = [pos0]
    velRef = [vel0]
    r = 0.1
    period = 10

    for k in range(N):
        predictTime = currentTime + k * sampleTime
        posX = r * np.sin(2.0 * np.pi / period * predictTime)
        posY = r * np.cos(2.0 * np.pi / period * predictTime)
        velX = 2.0 * np.pi / period * r * np.cos(2.0 * np.pi / period * predictTime)
        velY = - 2.0 * np.pi / period * r * np.sin(2.0 * np.pi / period * predictTime)
 
        posRef.append(startPos + np.array([0.0, posX, posY]).reshape(-1,1))
        velRef.append(np.array([0.0, velX, velY]).reshape(-1,1))
    
    # (-1, N+1)
    stateRef = np.concatenate((np.squeeze(np.array(posRef).T), np.squeeze(np.array(velRef).T)))
    
    currAccX = -(2.0 * np.pi / period)**2 * r * np.sin(2.0 * np.pi / period * currentTime)
    currAccY = -(2.0 * np.pi / period)**2 * r * np.cos(2.0 * np.pi / period * currentTime)
    currAccRef = np.array([0.0, currAccX, currAccY]).reshape(-1,1)
    return stateRef, currAccRef


class dataRecord:
    def __init__(self):
        self.xHistory = []
        self.uHistory = []
        self.tHistory = []
        self.xRefHistory = []
        self.zHistory = []
        self.objHistory = []

    def updateHistory(self, x, u, t, xRef, z, obj_val):
        self.xHistory.append(x)
        self.uHistory.append(u)
        self.tHistory.append(t)
        self.xRefHistory.append(xRef)
        self.zHistory.append(z)
        self.objHistory.append(obj_val)
        

    def plotData(self):
        stateMat = np.squeeze(np.array(self.xHistory))
        stateRefMat = np.squeeze(np.array(self.xRefHistory))
        controlMat = np.squeeze(np.array(self.uHistory))
        timeMat = np.squeeze(np.array(self.tHistory))
        zMat = np.squeeze(np.array(self.zHistory))
        objMat = np.squeeze(np.array(self.objHistory))

        if Is_passive:
            pasID = "Passive"
        else:
            pasID = "NoPassive"

        if Is_sim:
            simID = "Sim"
        else:
            simID = "Real"
        

        np.savetxt("data/"+simID+"RobotTrajectory"+pasID+timeID+".txt", stateMat)
        np.savetxt("data/"+simID+"RefTrajectory"+pasID+timeID+".txt", stateRefMat)
        np.savetxt("data/"+simID+"MPCCommand"+pasID+timeID+".txt", controlMat)
        np.savetxt("data/"+simID+"Obj"+pasID+timeID+".txt", objMat)
        state_dim = 3

        fig, ax = plt.subplots(state_dim, 1)
        for i in range(state_dim):
            ax[i].plot(stateMat[:, i])

        plt.figure()
        plt.plot(stateRefMat[:,1], stateRefMat[:,2], linestyle='--')
        plt.plot(stateMat[:,1], stateMat[:,2])

        fig, ax = plt.subplots(state_dim, 1)
        for i in range(state_dim):
            ax[i].plot(controlMat[:, i])

        fig, ax = plt.subplots(4, 1)
        ax[0].plot(zMat[:, 0])
        ax[1].plot(zMat[:, 1])
        ax[2].plot(zMat[:, 2])
        ax[3].plot(zMat[:, -1])
        ax[3].legend("z")

        errorMat = stateMat - stateRefMat
        error_yz = errorMat[5500:5900,1:3]
        a = np.sum(np.linalg.norm(error_yz, axis=1))*0.001
        print(pasID+' Energy is: ', a)

        return True


u_mpc = np.zeros((3,1))
prev_u_mpc = np.zeros((3,1))
def mpc_callback(msg):
    global u_mpc, z, obj_val

    z = copy.deepcopy(np.array(msg.position).reshape((-1,1)))
    obj_val = copy.deepcopy(np.array(msg.acceleration).reshape((-1,1)))

    if (len(msg.effort) != 3):
        print("mpc controller are not of size 3")
        u_mpc = copy.deepcopy(prev_u_mpc.reshape((-1,1)))
    else:
        u_mpc = copy.deepcopy(np.array(msg.effort).reshape((-1,1)))
        u_prev_u_mpc = u_mpc
        #print()


def quatdiff_in_euler(quat_curr, quat_des):
    """
        Compute difference between quaternions and return 
        Euler angles as difference
    """
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
        
    return -des_mat.dot(vec)

def control_thread(rate, start_pos):
    """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
    """
    simTime = 10
    beginTime = time.time()
    mpciter = 0
    last_J = robot.zero_jacobian()
    lastTime = rospy.get_rostime().to_sec()
    while not rospy.is_shutdown() and mpciter * sampleTime <= simTime:
            nowTime = rospy.get_rostime().to_sec()
            print("time:", nowTime-lastTime)
            lastTime = nowTime
            currentTime = mpciter * sampleTime
            
            #goal_pos = np.array([0.43562025, -0.0315768, 0.49585179]).reshape((-1,1))
            #goal_vel = np.zeros((3,1))
            # when using the panda_robot interface, the next 2 lines can be simplified 
            # to: `curr_pos, curr_ori = panda.ee_pose()`
            
            curr_pos = robot.endpoint_pose()['position'].reshape((3,1))
            curr_ori = np.asarray(robot.endpoint_pose()['orientation'])
            curr_vel = robot.endpoint_velocity()['linear'].reshape((3,1))
            curr_omg = robot.endpoint_velocity()['angular'].reshape((3,1))

            
            stateRef, currAccRef = circle_state_ref(mpciter, currentTime, curr_pos, curr_vel, start_pos)

            goal_pos = stateRef[0:3,1].reshape((3,1))
            delta_pos = goal_pos - curr_pos
            delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape((3,1))

            # when using the panda_robot interface, the next 2 lines can be simplified 
            # to: `curr_vel, curr_omg = panda.ee_velocity()`
            goal_vel = stateRef[3:6,1].reshape((3,1))
            delta_vel = goal_vel - curr_vel
            delta_omg = np.zeros((3,1)) - curr_omg
            # Update the system state
            
            state = np.concatenate((curr_pos, curr_vel), axis=0)
            dataRecord.updateHistory(state, u_mpc, time.time(), stateRef[:,1], z, obj_val)

            #q = list(robot.joint_angles().values())
            dq = list(robot.joint_velocities().values())
            # aarray = (ctypes.c_double*len(q))(*q)
            # barray = (ctypes.c_double*len(dq))(*dq)
            # c = [0.0 for i in range(49)]
            # carray = (ctypes.c_double*len(c))(*c)
            # so_coriolis.get_CoriolisMatrix(aarray, barray, carray)
            # ca_coriolis = np.array(carray).reshape((7,7))
            # # so_mass.get_MassMatrix(aarray, carray)
            # ca_mass = np.array(carray).reshape((7,7))
            # d = [0.0 for i in range(7)]
            # darray = (ctypes.c_double*len(d))(*d)
            # so_friction.get_FrictionTorque(aarray, darray)
            # ca_friction = np.array(darray).reshape((7,1))
            # so_gravity.get_CoriolisMatrix(aarray, barray)
            # ca_gravity = np.array(barray).reshape((7,1))

            # calculate cartesian mass matrix and coloris matrix
            J = robot.zero_jacobian()
            pinv_J = np.linalg.pinv(J)
            dot_J = (J - last_J)/sampleTime
            last_J = J
            
            # Desired task-space force using Impedance PD law
            # u = des_acc + M_v^-1(D_v(des_vel-curr_vel)+K_v(des_pos-curr_pos)-f))

            curr_joint_vel = np.array(dq).reshape((-1,1))

            inter_force = np.zeros((6,1))
            if currentTime >= 5.0 and currentTime < 5.5:
                inter_force[1,0] = 3.0
                inter_force[2,0] = 3.0
            
            #time-variant coffecient of impedance
            P_coe = (70. + currentTime * 3.)/90.
            if(P_coe>110/70): P_coe = 110/90

            #P_coe = 1.

            inv_Mv = np.linalg.inv(np.eye(6))
            PD = np.vstack([np.dot(P_coe*P_pos,(delta_pos)), P_ori*(delta_ori)]) + \
                np.vstack([np.dot(D_pos,(delta_vel)), D_ori*(delta_omg)])
            u = np.vstack([currAccRef, np.zeros((3,1))]) + 1.0 * np.dot(inv_Mv, \
                1.0 * inter_force + PD + 1.0 * np.vstack([u_mpc, np.zeros((3,1))]))
            
            v = np.dot(pinv_J, (u - np.dot(dot_J, curr_joint_vel)))
            # ca_mass, np.dot(ca_coriolis, curr_joint_vel)
            tau = np.dot(robot.joint_inertia_matrix(), v)+robot.coriolis_comp()[:,np.newaxis]

            # command robot using joint torques
            # panda_robot equivalent: panda.exec_torque_cmd(tau)
            robot.set_joint_torques(dict(list(zip(robot.joint_names(), tau))))
            # publish message to MPC regulator
            # command_msg.position -> actual_error
            # command_msg.velocity -> reference_trajectory
            # command_msg.acceleration -> stiffness
            # command_msg.effort -> force
            command_msg.position = np.hstack((np.squeeze(delta_pos), np.squeeze(delta_vel)))
            command_msg.velocity = np.hstack((np.squeeze(goal_pos), np.squeeze(goal_vel)))
            command_msg.acceleration = P_coe * np.diagonal(P_pos)
            command_msg.effort = inter_force
            pub.publish(command_msg)

            error = np.vstack((delta_pos, delta_vel))
    
            #print("mpciter", mpciter, "error:", error, "control_MPC:", u_mpc, "control_PD:", PD)
            mpciter = mpciter + 1
            rate.sleep()
            print("mpciter", mpciter, "error:", error, "control_MPC:", u_mpc, "control_PD:", PD)


def _on_shutdown():
    """
        Clean shutdown controller thread when rosnode dies.
    """
    global ctrl_thread, sub, pub
    if ctrl_thread.is_alive():
        ctrl_thread.join()
    
    sub.unregister()
    pub.unregister()

if __name__ == '__main__':

    rospy.init_node('mpc_test')
    # when using franka_ros_interface, the robot can be controlled through and
    # the robot state is directly accessible with the API
    # If using the panda_robot API, this will be
    # panda = PandaArm()
    sub = rospy.Subscriber('/Panda_simulator/own_controller/Joint_Commands', JointCommand, mpc_callback)
    
    command_msg = JointCommand()
    pub = rospy.Publisher('/Panda_simulator/own_state/Joint_Commands',
                          JointCommand, queue_size=1, tcp_nodelay=True)

    robot = ArmInterface()
    robot.move_to_neutral()
    print('--------------------Move to neutral pos--------------------')

    # when using the panda_robot interface, the next 2 lines can be simplified 
    # to: `start_pos, start_ori = panda.ee_pose()`
    ee_pose = robot.endpoint_pose()
    start_pos, start_ori = ee_pose['position'].reshape([3,1]), ee_pose['orientation']
    start_vel = robot.endpoint_velocity()['linear'].reshape([3,1])

    goal_pos, goal_ori = start_pos, start_ori
    z = 0.0
    obj_val = 0.0

    # test reference trajectory
    # mpciter = 0
    # while mpciter <= 20:
    #     robot.move_to_cartesian_pose(np.squeeze(start_pos)+circle_pos(mpciter),start_ori)
    #     mpciter = mpciter + 1
    
    # move to start pos 
    start_pos_circle,temp = circle_state(0)
    robot.move_to_cartesian_pose(np.squeeze(start_pos)+start_pos_circle,start_ori)
    print('--------------------Move to start pos of trajectory--------------------')
   
    # initialize MPC controller
    start_x = np.concatenate((start_pos, start_vel), axis=0)
    dataRecord = dataRecord()

    # start controller thread
    rospy.on_shutdown(_on_shutdown)
    rate = rospy.Rate(publish_rate)
    ctrl_thread = threading.Thread(target=control_thread, args = [rate, start_pos])
    ctrl_thread.start()

    rospy.spin()

    dataRecord.plotData()
    plt.show()
