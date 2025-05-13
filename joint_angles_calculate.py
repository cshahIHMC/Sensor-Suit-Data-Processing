import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
import Utility


def set_limb_structure():
    
    limb_structure = {
        "pelvis_l": ("pelvis", "pelvis_l"),
        "pelvis_r": ("pelvis", "pelvis_r"),
        "thigh_l": ("pelvis_l", "thigh_l"),
        "thigh_r": ("pelvis_r", "thigh_r"),
        "shank_l": ("thigh_l", "shank_l"),
        "shank_r": ("thigh_r", "shank_r"),
        "foot_l": ("shank_l", "foot_l"),
        "foot_r": ("shank_r", "foot_r"),
        "back": ("pelvis", "back")
    }
    
    # Child-parent joint combo
    joint_heirarchy = {
        "pelvis" : "pelvis",
        "thigh_r" : "pelvis",
        "thigh_l" : "pelvis",
        "shank_r" : "thigh_r",
        "shank_l" : "thigh_l",
        "foot_r" : "shank_r",
        "foot_l" : "shank_l"
        
    }
    
    return limb_structure, joint_heirarchy

def get_joint_imu_map():
    
    joint_imu_map = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu5_quat", 
        "shank_r": "imu6_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu4_quat",
        # "back": "imu3_quat",
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    
    joint_imu_map_microstrain = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu5_quat", 
        "shank_r": "imu6_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu4_quat",
        # "back": "imu3_quat",
    }
    
    joint_imu_map_insole = {
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    return joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole

# Apply transformations to the quaternions
# - Zero with Pelvis
# - Bring them to animation frame
def transform_quaternions(data, t_pose_q, transforms):
    
    transformed_data = {}
    quat_norm = {}
    
    # Get the t pose quat and normalize them
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}
    
    # Loop over all IMU's and according to their frames apply the correct transformations
    for imu in data.keys():
        
        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[imu], axis=1, keepdims=True)
        quat_norm[imu] = R.from_quat(data[imu]/quat_magnitude, scalar_first=True)
        
        
        # For the case of the locked pelvis
        if "foot" in imu:
            quat_norm[imu] = t_pose_q_norm["pelvis"] * transforms["pelvis_2_foot"] * quat_norm[imu] * t_pose_q_norm[imu].inv() * transforms["pelvis_2_foot"].inv()
        
        
        relative_rotations = transforms["Anatomical_2_pelvis"] * quat_norm["pelvis"].inv() * quat_norm[imu] * transforms["Anatomical_2_pelvis"].inv()
        
        
        # Stored the transformed_data
        transformed_data[imu] = relative_rotations
        
    return transformed_data

# Calculate the joint angles using the transformed quaternion data
def cal_joint_angles(quaternion_data, joint_heirarchy, transforms):
    joint_quaternions = {}
    joint_angles = {}
    
    
    # Iterate over all imu's in joint heirarchy
    for child, parent in joint_heirarchy.items():
        
        print("Child ", child)
        print("Parent ", parent)
        
        joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
        # joint_quaternions[child] = quaternion_data[child]
        
        # if "foot" in child:
        #     joint_quaternions[child] = quaternion_data[child]
        # else:
        #     joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
            
            
        
        
        # joint_angles[child] = joint_quaternions[child].as_euler('xyz', degrees=True)
        
        rotvec = joint_quaternions[child].as_rotvec()
        
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)

        if angle_rad > 1e-8:
            axis = rotvec / angle_rad

            # Project total rotation onto anatomical axes
            angle_x = angle_deg * np.dot(axis, [1, 0, 0])
            angle_y = angle_deg * np.dot(axis, [0, 1, 0])
            angle_z = angle_deg * np.dot(axis, [0, 0, 1])
        else:
            # If the rotation is ~0, just zero everything
            axis = np.array([0.0, 0.0, 0.0])
            angle_x = angle_y = angle_z = 0.0

        # Store in your joint_angles dict
        joint_angles[child] = {
            "angle_deg": angle_deg,     # Total rotation
            "axis": axis,               # Rotation axis (unit vector)
            "angle_x": angle_x,         # Component of rotation about X
            "angle_y": angle_y,         # Component of rotation about Y
            "angle_z": angle_z          # Component of rotation about Z
        }
        
    return joint_angles

def plot_joint_angles(joint_angles, GRF):
    
    fig, axes = plt.subplots(4, len(list(joint_angles.keys())), figsize=(20,12), sharex=True)
    
    print(joint_angles.keys())
    
    lim = 100
    
    for i, joint in enumerate(joint_angles.keys()):
        
        joint_angle = joint_angles[joint]
        
        # Plot X axis
        axes[0,i].plot(joint_angle["angle_x"], linewidth=1, color="red")
        axes[0,i].set_title(joint + "_X (deg)")
        
        # Plot Y axis
        axes[1,i].plot(joint_angle["angle_y"], linewidth=1, color="green")
        axes[1,i].set_title(joint + "_Y (deg)")
    
        # Plot Z axis
        axes[2,i].plot(joint_angle["angle_z"], linewidth=1, color="blue")
        axes[2,i].set_title(joint + "_Z (deg)")
        
        if "_r" in joint:
            axes[3,i].plot(GRF["R_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Right GRF")
        else:
            axes[3,i].plot(GRF["L_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Left GRF")
            
        axes[0,i].set_ylim(-lim,lim)
        axes[1,i].set_ylim(-lim,lim)
        axes[2,i].set_ylim(-lim,lim)
        # axes[3,i].set_ylim(-lim,lim)
        
        
    fig.suptitle("Joint Angles(Deg)")
    
    plt.tight_layout()
    plt.show()
    
        
    

def main():
    
    csv_path = "/home/cshah/workspaces/sensorsuit/logs/05_08_2025/05_08_2025_start_0_walk_test.csv"
    # Load the data to a csv
    data = Utility.load_quaternion_data(csv_path=csv_path)
    
    force = ["L_insole_force", "R_insole_force"]
    insole_force_data = data[force]
    
    
    # Build the Transforms from one frame to the other
    body_transforms = Utility.build_transforms_2()
    limb_structure, joint_heirarchy = set_limb_structure()
    joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole = get_joint_imu_map()
    
    quaternion_data, t_pose_quat = Utility.extract_data(data=data, 
                                                        joint_imu_map_microstrain=joint_imu_map_microstrain, 
                                                        joint_imu_map_insole=joint_imu_map_insole)
    

    # Transform the quaternions - Zero them to the pelvis frame and transform to animation frame
    transformed_quat = transform_quaternions(data=quaternion_data, t_pose_q=t_pose_quat, transforms=body_transforms)
    
    # Get the joint angles
    joint_angles = cal_joint_angles(transformed_quat, joint_heirarchy, body_transforms)
    

    # Plot the joint angles
    plot_joint_angles(joint_angles, insole_force_data)
    




if __name__ == "__main__":
    raise SystemExit(main())
    