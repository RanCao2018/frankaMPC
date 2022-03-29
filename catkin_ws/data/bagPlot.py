from cProfile import label
import sys
import rosbag
import matplotlib.pyplot as plt
import numpy as np

##################
# DESCRIPTION:
# plot the robot states from a rosbag (for visualization)
# 
# USAGE EXAMPLE:
# python bagPlot.py /root/catkin_ws/bagfiles your_bagfile.bag
# ##################

filename = sys.argv[2]
directory = sys.argv[1]
print("Reading the rosbag file")
if not directory.endswith("/"):
  directory += "/"
extension = ""
if not filename.endswith(".bag"):
  extension = ".bag"
bag = rosbag.Bag(directory + filename + extension)

print("Plotting robot state data")

reference_position_y_list = []
reference_position_z_list = []
# Get all message on the states topic
for topic, msg, t in bag.read_messages(topics=['/complete_cartesian_impedance_example_controller/equilibrium_pose']):
  # Only write to CSV if the message is for our robot
    reference_position_y_list.append(msg.pose_d.pose.position.y)
    reference_position_z_list.append(msg.pose_d.pose.position.z)
reference_position_y = np.array(reference_position_y_list)
reference_position_z = np.array(reference_position_z_list)

position_y_list = []
position_z_list = []
# Get all message on the states topic
for topic, msg, t in bag.read_messages(topics=['/franka_state_controller/franka_states']):
  # Only write to CSV if the message is for our robot
    position_y_list.append(msg.O_T_EE[13])
    position_z_list.append(msg.O_T_EE[14])
position_y = np.array(position_y_list)
position_z = np.array(position_z_list)


print(len(reference_position_y))
print(len(position_y))

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(reference_position_y, reference_position_z, '--', color='red', linewidth=2, label='reference trajectory')
ax.plot(position_y, position_z, '-', color='blue', linewidth=2, label='real trajectory')
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.legend()
plt.show()

bag.close()