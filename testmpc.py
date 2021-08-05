#!/usr/bin/env python
import casadi as ca
import casadi.tools as ca_tools

import matplotlib.pyplot as plt
import numpy as np
import time

import rospy
import copy
import threading
import quaternion
from franka_interface import ArmInterface

# --------- Modify as required ------------
# MPC controller parameters
N = 8
sampleTime = 0.5
Q = np.eye(6)
R = 0.1*np.eye(3)
R[1, 1] = 0.01
# -----------------------------------------
publish_rate = 100
# -----------------------------------------

def shift_movement(T, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value.full()
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return st, u_end, x_f


def desired_command_and_trajectory(t, sample_time, x0_, N_):
    # initial state / last state
    x_ = []
    x_.append(x0_.reshape(1, -1))
    u_ = []
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + sample_time*i
        x_ref_ = 0.0 * t_predict * np.ones((1, 6))
        u_ref_ = 0 * np.ones((1, 3))
        x_ref_[:, 1] = 0.5
        x_.append(x_ref_)
        u_.append(u_ref_)
    # return pose and command
    x_ = np.array(x_).reshape(N_+1, -1)
    u_ = np.array(u_).reshape(N, -1)
    return x_, u_


def get_estimated_result(data, N_):
    x_ = np.zeros((6, N_+1))
    u_ = np.zeros((3, N_))
    for i in range(N_):
        x_[:, i] = np.squeeze(data[i*9:i*9+6])
        u_[:, i] = np.squeeze(data[i*9+6:i*9+9])
    x_[:, -1] = np.squeeze(data[-6:])
    return u_, x_


class MPCController:
    def __init__(self, N, sampleTime, Q, R, x=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1), u = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)):
        self.simTime = 30.0
        self.N = N
        self.sampleTime = sampleTime
        self.Q = Q
        self.R = R
        self.x = x
        self.u = u

        self.predictControlHistory = []
        self.predictStateHistory = []
        self.objValueHistory = []
        self.xHistory = []
        self.uHistory = []
        self.tHistory = []

        # Define state variable
        states = ca_tools.struct_symSX([
            (
                ca_tools.entry('e', shape=(3, 1)),
                ca_tools.entry('de', shape=(3, 1)),
            )
        ])
        e, de = states[...]
        self.n_states = states.size
        # Define control input
        controls = ca_tools.struct_symSX(
            [(ca_tools.entry('u', shape=(3, 1)),)])
        u, = controls[...]
        self.n_controls = controls.size
        # define parameter matices
        A1 = np.eye(3)
        B = np.eye(3)
        C = np.eye(3)

        rhs = ca_tools.struct_SX(states)
        rhs['e'] = de
        rhs['de'] = - ca.mtimes(A1, e) - ca.mtimes(B, u)
        self.f = ca.Function('f', [states, controls], [rhs])

        # Define MPC Problem
        self.optimizing_target = ca_tools.struct_symSX([
            (
                ca_tools.entry('X', repeat=N+1, struct=states),
                ca_tools.entry('U', repeat=N, struct=controls)
            )
        ])
        # data are stored in list [], notice that ',' cannot be missed
        X, U, = self.optimizing_target[...]
        self.current_parameters = ca_tools.struct_symSX([
            (
                ca_tools.entry('X_ref', repeat=N+1, struct=states),
                ca_tools.entry('U_ref', repeat=N, struct=controls),
            )
        ])
        X_ref, U_ref, = self.current_parameters[...]

        obj = 0
        g = []
        g.append(X[0] - X_ref[0])
        for i in range(N):
            state_error_ = X[i] - X_ref[i+1]
            control_error_ = U[i] - U_ref[i]
            obj = obj + ca.mtimes([state_error_.T, self.Q, state_error_]) + \
                ca.mtimes([control_error_.T, self.R, control_error_])
            x_next_ = self.f(X[i], U[i]) * sampleTime + X[i]
            g.append(X[i+1] - x_next_)

        nlp_prob = {'f': obj, 'x': self.optimizing_target,
                    'p': self.current_parameters, 'g': ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        lbg = 0.0
        ubg = 0.0
        lbx = []
        ubx = []
        for _ in range(N):
            lbx = lbx + [-np.inf, -np.inf, -np.inf, -np.inf, -
                         np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
            ubx = ubx + [np.inf, np.inf, np.inf, np.inf,
                         np.inf, np.inf, np.inf, np.inf, np.inf]
        # for the N+1 state terminal constraint
        lbx = lbx + [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        ubx = ubx + [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        self.solArgs = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }

        self.referSignal = self.current_parameters(0)  # Reference Signal
        self.initVar = self.optimizing_target(0)  # Decision Variable

    def rollout(self, xRef, uRef, xPredict, uPredict):
        lbg = self.solArgs['lbg']
        ubg = self.solArgs['ubg']
        lbx = self.solArgs['lbx']
        ubx = self.solArgs['ubx']

        self.referSignal['X_ref', lambda x:ca.horzcat(*x)] = xRef.T
        self.referSignal['U_ref', lambda x:ca.horzcat(*x)] = uRef.T
        self.initVar['X', lambda x:ca.horzcat(*x)] = xPredict
        self.initVar['U', lambda x:ca.horzcat(*x)] = uPredict

        sol = self.solver(x0=self.initVar, p=self.referSignal,
                          lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

        #  predictControl = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        #  predictState = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
        predictControl, predictState = get_estimated_result(sol['x'].full(), N)
        self.predictControlHistory.append(predictControl)
        self.predictStateHistory.append(predictState)
        self.objValueHistory.append(sol['f'].full())

        return predictControl, predictState

    def updateHistory(self, x, u, t):
        self.xHistory.append(x)
        self.uHistory.append(u)
        self.tHistory.append(t)

    def testExample(self):
        # MPC Start
        simTime = 30.0

        xPredict = np.tile(self.x, (1, N+1))
        uPredict = np.tile(self.u, (1, N))

        mpciter = 0
        costTime = time.time()
        while(mpciter * sampleTime < simTime):
            currentTime = mpciter * sampleTime
            xRef, uRef = desired_command_and_trajectory(
                currentTime, self.sampleTime, x, self.N)
            predictControl, predictState = self.rollout(
                xRef, uRef, xPredict, uPredict)

            # True System
            x, uPredict, xPredict = shift_movement(
                self.sampleTime, x, predictControl, predictState, self.f)
            u = np.reshape(
                np.array(predictControl[:, 0]), (self.n_controls, 1))
            self.updateHistory(x, u, currentTime)

            mpciter = mpciter + 1

        print(time.time()-costTime)
        return True

    def plotMPC(self):
        stateMat = np.squeeze(np.array(self.xHistory))
        controlMat = np.squeeze(np.array(self.uHistory))
        timeMat = np.squeeze(np.array(self.tHistory))
        state_dim = 3

        fig, ax = plt.subplots(state_dim, 1)
        for i in range(state_dim):
            ax[i].plot(stateMat[:, i])

        fig, ax = plt.subplots(state_dim, 1)
        for i in range(state_dim):
            ax[i].plot(controlMat[:, i])

        return True

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

def control_thread(rate, MPC):
    """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
    """
    while not rospy.is_shutdown():
        error = 100.0
        u_MPC = np.zeros((3,1))
        while error > 0.005:
            # when using the panda_robot interface, the next 2 lines can be simplified 
            # to: `curr_pos, curr_ori = panda.ee_pose()`
            curr_pos = robot.endpoint_pose()['position'].reshape([3,1])
            curr_ori = np.asarray(robot.endpoint_pose()['orientation'])

            delta_pos = (goal_pos - curr_pos).reshape([3,1]).reshape([3,1])
            delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3,1])

            # when using the panda_robot interface, the next 2 lines can be simplified 
            # to: `curr_vel, curr_omg = panda.ee_velocity()`
            curr_vel = robot.endpoint_velocity()['linear'].reshape([3,1])
            curr_omg = robot.endpoint_velocity()['angular'].reshape([3,1])
            
            goal_vel = np.zeros((3,1))
            # calculate MPC ouput

            xRef = np.concatenate((goal_pos, goal_vel), axis=0)
            uRef = np.zeros((3,1))
            x = np.concatenate((curr_pos, curr_vel), axis=0)
            xPredict = np.tile(x, (1, N+1))
            uPredict = np.zeros((3, N))

            MPC.updateHistory(x, u_MPC, time.time())
            predictControl, predictState = MPC.rollout(
                xRef, uRef, xPredict, uPredict)
            
            u_MPC = np.reshape(
                np.array(predictControl[:, 0]), (3, 1))

            # for test

            u_MPC = np.zeros((3,1))

            # Desired task-space force using PD law
            F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
                np.vstack([D_pos*(curr_vel), D_ori*(curr_omg)]) + u_MPC

            error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)
            
            # panda_robot equivalent: panda.jacobian(angles[optional]) or panda.zero_jacobian()
            J = robot.zero_jacobian()
            
            # joint torques to be commanded
            tau = np.dot(J.T,F)

            # command robot using joint torques
            # panda_robot equivalent: panda.exec_torque_cmd(tau)
            robot.set_joint_torques(dict(list(zip(robot.joint_names(), tau))))
            
            rate.sleep()

def _on_shutdown():
    """
        Clean shutdown controller thread when rosnode dies.
    """
    global ctrl_thread
    if ctrl_thread.is_alive():
        ctrl_thread.join()

if __name__ == '__main__':

    rospy.init_node('mpc_test')
    # when using franka_ros_interface, the robot can be controlled through and
    # the robot state is directly accessible with the API
    # If using the panda_robot API, this will be
    # panda = PandaArm()
    robot = ArmInterface()

    # when using the panda_robot interface, the next 2 lines can be simplified 
    # to: `start_pos, start_ori = panda.ee_pose()`
    ee_pose = robot.endpoint_pose()
    start_pos, start_ori = ee_pose['position'].reshape([3,1]), ee_pose['orientation']
    start_vel = robot.endpoint_velocity()['linear'].reshape([3,1])

    goal_pos, goal_ori = start_pos, start_ori
    # initialize MPC controller
    start_x = np.concatenate((start_pos, start_vel), axis=0)
    MPC = MPCController(N, sampleTime, Q, R, start_x)

    # start controller thread
    rospy.on_shutdown(_on_shutdown)
    rate = rospy.Rate(publish_rate)
    ctrl_thread = threading.Thread(target=control_thread, args = [rate, MPC])
    ctrl_thread.start()

    rospy.spin()

    MPC.plotMPC()
    plt.show()
