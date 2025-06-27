import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
import Utility

from scipy.signal import butter, filtfilt


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
        "thigh_r": "imu1_quat", 
        "shank_r": "imu6_quat", # 06/17 - imu5, 06/16 - imu6
        "thigh_l": "imu5_quat",
        "shank_l": "imu4_quat",# 06/17 - imu4, 06/16 - imu4
        # "back": "imu3_quat",
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    
    joint_imu_map_microstrain = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu1_quat", 
        "shank_r": "imu6_quat",
        "thigh_l": "imu5_quat",
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
    
    gravity_vector = [0,0,-1]
    
    # Get the t pose quat and normalize them
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}
    
    
    
    # Step 1: Compute pelvis gravity alignment
    pelvis_rot = t_pose_q_norm["pelvis"]
    pelvis_z = pelvis_rot.apply([0, 0, 1])
    q_gravity_align_pelvis = R.align_vectors([gravity_vector], [pelvis_z])[0]
    q_pelvis_anatomical = q_gravity_align_pelvis * pelvis_rot
    
    
    # t_pose_q_norm = t_pose_q
    # Loop over all IMU's and according to their frames apply the correct transformations
    for imu in data.keys():
        
        print(imu)
        
        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[imu], axis=1, keepdims=True)
        quat_norm[imu] = R.from_quat(data[imu]/quat_magnitude, scalar_first=True)
                
        # Step 1: Gravity alignment for T-pose
        t_pose_rot = t_pose_q_norm[imu]
        t_pose_z = t_pose_rot.apply([0, 0, 1])
        
        q_gravity_align = R.align_vectors([gravity_vector], [t_pose_z])[0]
        
        # Step 2: Pelvis anatomical frame (after gravity alignment)
        q_anatomical = q_gravity_align * t_pose_rot
        
        # Step 3: Correction quaternion (sensor-to-anatomical)
        q_segment_correction = q_pelvis_anatomical.inv() * q_anatomical
        
        # print(q_correction.as_euler("xyz", degrees=True))
      
        
       
    
        frame_name = "Anatomical_2_" + imu
        
        # relative_rotations = transforms[frame_name] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name].inv()
        
      
        
        if "foot_r" in imu:
            
            
            
            # Toe Transform
            # toe_transform_r = ( quat_norm[imu] * quat_norm["shank_r"].inv() ).as_quat()
            # toe_transform_r = np.mean(toe_transform_r, axis=0)
            # print(toe_transform_r)
            
            toe_transform_r_1 = [ 0.57662261, -0.81659031,  0.00673089,  0.0213726 ]
             
            # 10min in Calibration
            transform_r = [ 0.96805752,  0.24972168, -0.01774981,  0.01163004]
            relative_rotations = R.from_quat(toe_transform_r_1) * quat_norm[imu]
            # relative_rotations = quat_norm[imu]
        elif "foot_l" in imu:
            
            # Toe Transform
            # toe_transform_l = ( quat_norm[imu] * quat_norm["shank_l"].inv() ).as_quat()
            # toe_transform_l = np.mean(toe_transform_l, axis=0)
            # print(toe_transform_l)
            
            toe_transform_l_1 = [ 0.40604399, -0.91360987,  0.0097929,   0.0145006 ]
            
            # 10min in Calibration
            transform_l = [ 0.21206732,  0.97710702, -0.01205927, -0.00935059]
 
            # relative_rotations =      (R.from_quat(transform_l) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_l) * quat_norm[imu]
            relative_rotations = R.from_quat(toe_transform_l_1) * quat_norm[imu]        
        else:
            # relative_rotations = t_pose_q_norm[imu].inv() * quat_norm[imu] 
            relative_rotations = quat_norm[imu] 
        
        
        # relative_rotations = q_segment_correction * relative_rotations
        
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
        
        joint_quaternions[child] = quaternion_data[child]
        # joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
        
        # if "foot" in child:
        #     joint_quaternions[child] = quaternion_data[child]
        # else:
            # joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
            
            
        
        
        euler_angles = joint_quaternions[child].as_euler('zyx', degrees=True)

        rotvec = joint_quaternions[child].as_rotvec()
        
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)

        # if angle_rad > 1e-8:
        #     axis = rotvec / angle_rad

        #     # Project total rotation onto anatomical axes
        #     angle_x = angle_deg * np.dot(axis, [1, 0, 0])
        #     angle_y = angle_deg * np.dot(axis, [0, 1, 0])
        #     angle_z = angle_deg * np.dot(axis, [0, 0, 1])
        # else:
        #     # If the rotation is ~0, just zero everything
        #     axis = np.array([0.0, 0.0, 0.0])
        #     angle_x = angle_y = angle_z = 0.0
        
        angle_x = euler_angles[:, 0]
        angle_y = euler_angles[:, 1]
        angle_z = euler_angles[:, 2]

        # Store in your joint_angles dict
        joint_angles[child] = {
            "angle_deg": angle_deg,     # Total rotation
            # "axis": axis,               # Rotation axis (unit vector)
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
        
        if i == 0:
            continue
        
        if "thigh_r" in joint:
            joint_name = "hip_r_angle"
        elif "thigh_l" in joint:
            joint_name = "hip_l_angle"
        elif "shank_r" in joint:
            joint_name = "knee_r_angle"
        elif "shank_l" in joint:
            joint_name = "knee_l_angle"
        elif "foot_r" in joint:
            joint_name = "ankle_r_angle"
        elif "foot_l" in joint:
            joint_name = "ankle_l_angle"
            
        
        # Plot X axis
        axes[0,i].plot(joint_angle["angle_x"], linewidth=1, color="red")
        axes[0,i].set_title(joint_name + "_X (deg)")
        
        # Plot Y axis
        axes[1,i].plot(joint_angle["angle_y"], linewidth=1, color="green")
        axes[1,i].set_title(joint_name + "_Y (deg)")
    
        # Plot Z axis
        axes[2,i].plot(joint_angle["angle_z"], linewidth=1, color="blue")
        axes[2,i].set_title(joint_name + "_Z (deg)")
        
        # if "_r" in joint:
        #     axes[3,i].plot(GRF["R_insole_force"]/50, linewidth=1, color="black")
        #     axes[3,i].set_title("Right GRF")
        # else:
        #     axes[3,i].plot(GRF["L_insole_force"]/50, linewidth=1, color="black")
        #     axes[3,i].set_title("Left GRF")
            
        # axes[0,i].set_ylim(-lim,lim)
        # axes[1,i].set_ylim(-lim,lim)
        # axes[2,i].set_ylim(-lim,lim)
        # axes[3,i].set_ylim(-lim,lim)
        
        
    fig.suptitle("Joint Angles(Deg)")
    
    plt.tight_layout()
    plt.show()
    
        
    

def main():
    
    csv_path = "/home/cshah/workspaces/sensorsuit/logs/06_16_stand_turn_stand.csv"

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
    