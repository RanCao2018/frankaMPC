import sys
import os
import csv
import rosbag
import rospy

##################
# DESCRIPTION:
# Creates CSV files of the robot joint states from a rosbag (for visualization with e.g. pybullet)
# 
# USAGE EXAMPLE:
# rosrun your_package get_jstate_csvs.py /root/catkin_ws/bagfiles your_bagfile.bag
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

# Create directory with name filename (without extension)
results_dir = directory + filename[:-4] + "_results"
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

print("Writing robot joint state data to CSV")

with open(results_dir +"/"+filename+'_states.csv', mode='w') as data_file:
  data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  data_writer.writerow(['time', 'position_x', 'position_y', 
  'position_z'])
  # Get all message on the /joint states topic
  for topic, msg, t in bag.read_messages(topics=['/complete_cartesian_impedance_example_controller/equilibrium_pose']):
    # Only write to CSV if the message is for our robot
      data_writer.writerow([t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
  # for topic, msg, t in bag.read_messages(topics=['/franka_state_controller/franka_states']):
  #   # Only write to CSV if the message is for our robot
  #     data_writer.writerow([t, msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])


print("Finished creating csv file!")
bag.close()