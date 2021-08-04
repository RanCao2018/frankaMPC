#!/usr/bin/env python
import casadi as ca 
import casadi.tools as ca_tools

import matplotlib.pyplot as plt
import numpy as np 
import time

def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f

def desired_command_and_trajectory(t, sample_time, x0_, N_):
    # initial state / last state
    x_ = []
    x_.append(x0_.reshape(1, -1))
    u_ = []
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + sample_time*i
        x_ref_ = 0.0 * t_predict * np.ones((1,6))
        u_ref_ = 0 * np.ones((1,3))
        x_ref_[:,1] = 0.5
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
    x_[:,-1] = np.squeeze(data[-6:])
    return u_, x_

class MPCController:
  def __init__(self, N, sampleTime, Q, R):
    self.simTime = 30.0
    self.N = N
    self.sampleTime = sampleTime
    self.Q = Q
    self.R = R

simTime = 30.0
N = 8
sampleTime = 0.5
# Define state variable
states = ca_tools.struct_symSX([
        (
            ca_tools.entry('e', shape= (3,1)),
            ca_tools.entry('de', shape = (3,1)),
        )
    ])
e, de = states[...]
n_states = states.size
# Define control input
controls  = ca_tools.struct_symSX([(ca_tools.entry('u', shape = (3,1)),)])
u, = controls[...]
n_controls = controls.size
# define parameter matices
A1 = np.eye(3)
B = np.eye(3)
C = np.eye(3)

rhs = ca_tools.struct_SX(states)
rhs['e'] = de
rhs['de'] = - ca.mtimes(A1, e) - ca.mtimes(B, u)
f = ca.Function('f', [states, controls], [rhs])

n_reference = n_states+n_controls

# Define MPC Problem
optimizing_target = ca_tools.struct_symSX([
        (
            ca_tools.entry('X', repeat=N+1, struct=states),
            ca_tools.entry('U', repeat=N, struct=controls) 
        )
])
X, U, = optimizing_target[...] # data are stored in list [], notice that ',' cannot be missed
current_parameters = ca_tools.struct_symSX([
        (
            ca_tools.entry('X_ref', repeat=N+1, struct=states),
            ca_tools.entry('U_ref', repeat=N, struct=controls),
        )
])
X_ref, U_ref, = current_parameters[...]

obj = 0
g = []
Q = np.eye(6)
R = 0.1*np.eye(3)
R[1,1] = 0.01

g.append(X[0] - X_ref[0])
for i in range(N):
  state_error_ = X[i] - X_ref[i+1]
  control_error_ = U[i] - U_ref[i]
  obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) + ca.mtimes([control_error_.T, R, control_error_])
  x_next_ = f(X[i], U[i]) * sampleTime + X[i]
  g.append(X[i+1] - x_next_)


nlp_prob = {'f': obj, 'x': optimizing_target, 'p':current_parameters, 'g':ca.vertcat(*g)}
opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

lbg = 0.0
ubg = 0.0
lbx = []
ubx = []
for _ in range(N):
  lbx = lbx + [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
  ubx = ubx + [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
# for the N+1 state terminal constraint
lbx = lbx + [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
ubx = ubx + [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

# MPC Start
t = 0.0
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
u = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
xPredict = np.tile(x, (1, N+1))
uPredict = np.tile(u, (1, N))
xHistory = [x]
tHistory = [t]
uHistory = [u]

predictStateHistory = []
predictControlHistory = []

referSignal = current_parameters(0) # Reference Signal
initVar = optimizing_target(0) # Decision Variable
mpciter = 0
costTime = time.time()
while(mpciter * sampleTime < simTime):
  currentTime = mpciter * sampleTime
  xRef, uRef = desired_command_and_trajectory(t, sampleTime, x, N)
  referSignal['X_ref', lambda x:ca.horzcat(*x)] = xRef.T
  referSignal['U_ref', lambda x:ca.horzcat(*x)] = uRef.T
  initVar['X', lambda x:ca.horzcat(*x)] = xPredict
  initVar['U', lambda x:ca.horzcat(*x)] = uPredict

  sol = solver(x0=initVar, p=referSignal, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

#   predictControl = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
#   predictState = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
  predictControl, predictState = get_estimated_result(sol['x'].full(), N)
  predictControlHistory.append(predictControl)
  predictStateHistory.append(predictState)
  print(sol['f'])

  # True System
  t, x, uPredict, xPredict = shift_movement(sampleTime, t, x, predictControl, predictState, f)
  u = np.reshape(np.array(predictControl[:, 0]), (n_controls, 1))
  xHistory.append(x)
  uHistory.append(u)
  tHistory.append(t)

  mpciter = mpciter + 1

print(time.time()-costTime)

stateMat = np.squeeze(np.array(xHistory))
controlMat = np.squeeze(np.array(uHistory))
timeMat = np.squeeze(np.array(tHistory))
state_dim = 3

fig, ax = plt.subplots(state_dim, 1)
for i in range(state_dim):
  ax[i].plot(stateMat[:,i])
plt.show()

fig, ax = plt.subplots(state_dim, 1)
for i in range(state_dim):
  ax[i].plot(controlMat[:,i])