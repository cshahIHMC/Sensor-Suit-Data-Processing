import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
import Utility
from scipy.signal import resample
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import butter, filtfilt
import csv


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
    
    # Joint IMU Map
    joint_imu_map = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu5_quat", 
        "shank_r": "imu4_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu6_quat",
        # "back": "imu3_quat",
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    
    joint_imu_map_microstrain = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu5_quat", 
        "shank_r": "imu4_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu6_quat",
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
def transform_quaternions(data, t_pose_q, transforms, joint_heirarchy):
    
    transformed_data = {}
    quat_norm = {}
    raw_quaternions = {}
    
    # Get the t pose quat and normalize them
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}

    
    # Loop over all IMU's and according to their frames apply the correct transformations
    # for imu in data.keys():
    for imu, parent in joint_heirarchy.items():
        
        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[imu], axis=1, keepdims=True)
        quat_norm[imu] = R.from_quat(data[imu]/quat_magnitude, scalar_first=True)

        if "foot_r" in imu:
            # 10min in Calibration
            transform_r = [ 0.96805752,  0.24972168, -0.01774981,  0.01163004]
            # relative_rotations =  (R.from_quat(transform_r) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_r) * quat_norm[imu] 
            A = R.from_quat(transform_r) * quat_norm[imu]
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_r) * t_pose_q_norm[imu]
            D = t_pose_q_norm["pelvis"].inv() * C
            relative_rotations = D.inv() * B 
            
        elif "foot_l" in imu:

            # 10min in Calibration
            transform_l = [ 0.21206732,  0.97710702, -0.01205927, -0.00935059]
 
            # relative_rotations =      (R.from_quat(transform_l) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_l) * quat_norm[imu] 
            A = R.from_quat(transform_l) * quat_norm[imu]
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_l) * t_pose_q_norm[imu]
            D = t_pose_q_norm["pelvis"].inv() * C
            relative_rotations =  D.inv() * B 
        else:
            # relative_rotations = t_pose_q_norm[imu].inv() * quat_norm[imu] 
            A = quat_norm["pelvis"].inv() * quat_norm[imu]
            B = t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[imu]
            
            relative_rotations = B.inv() * A 
             
        raw_quaternions[f"{imu}_ss"] = (relative_rotations[1359:7059])
        
        print(len(raw_quaternions[f"{imu}_ss"]))
        # Stored the transformed_data
        transformed_data[imu] = relative_rotations
        
    return transformed_data, raw_quaternions

# Calculate the joint angles using the transformed quaternion data
def cal_joint_angles(quaternion_data, joint_heirarchy, transforms, raw_quaternions):
    joint_quaternions = {}
    joint_angles = {}
    joint_rel_raw_quaternions = {}
    
    
    # Iterate over all imu's in joint heirarchy
    for child, parent in joint_heirarchy.items():
        
        # print("Child ", child)
        # print("Parent ", parent)
            
        
        joint_quaternions[child] = quaternion_data[child]
        
        if "pelvis" in child or "thigh" in child:
            joint_quaternions[child] = quaternion_data[child]
            joint_rel_raw_quaternions_imu = raw_quaternions[f"{child}_ss"].as_quat()
            
        else:
            joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
            
            joint_rel_raw_quaternions_imu = (raw_quaternions[f"{parent}_ss"].inv() * raw_quaternions[f"{child}_ss"]).as_quat()
            
        
        
        # Normalzing the Quaternions that are sent from the sensor suit side   
        joint_rel_raw_quaternions[f"{child}_ss"] = joint_rel_raw_quaternions_imu / np.linalg.norm(joint_rel_raw_quaternions_imu, axis=1, keepdims=True)
        
        # joint_angles[child] = joint_quaternions[child].as_euler('xyz', degrees=True)
        
        rotvec = joint_quaternions[child].as_rotvec()
        
        
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)

        # if angle_rad > 1e-8:
        #     axis = rotvec / angle_rad

        #     # Project total rotation onto anatomical axes
        #     # angle_x = angle_deg * np.dot(axis, [1, 0, 0])
        #     # angle_y = angle_deg * np.dot(axis, [0, 1, 0])
        #     # angle_z = angle_deg * np.dot(axis, [0, 0, 1])
            
        #     angle_x = joint_quaternions[child].as_quat()[:,0]
        #     angle_y = joint_quaternions[child].as_quat()[:,1]
        #     angle_z = joint_quaternions[child].as_quat()[:,2]
        # else:
        #     # If the rotation is ~0, just zero everything
            
        #     axis = np.array([0.0, 0.0, 0.0])
        #     angle_x = angle_y = angle_z = 0.0
        
        joint_angle_euler = joint_quaternions[child].as_euler("zyx", degrees = True)
        
        angle_x = joint_angle_euler[:, 0] 
        angle_y = joint_angle_euler[:, 1]
        angle_z = joint_angle_euler[:, 2]

        # Store in your joint_angles dict
        joint_angles[child] = {
            "angle_deg": angle_deg,     # Total rotation
            # "axis": axis,               # Rotation axis (unit vector)
            "angle_x": angle_x,         # Component of rotation about X
            "angle_y": angle_y,         # Component of rotation about Y
            "angle_z": angle_z          # Component of rotation about Z
        }
        
    return joint_angles, joint_rel_raw_quaternions

def re_structure_ss_df(ss_df):
    
    
    ss_new_df = pd.DataFrame()
    
    Joint_angles_col = ["hip_flexion_r",
                        "hip_rotation_r",
                        "hip_adduction_r",
                        "hip_flexion_l",
                        "hip_rotation_l",
                        "hip_adduction_l",
                        "knee_angle_r",
                        "knee_angle_l",
                        "ankle_angle_r",
                        "ankle_angle_l"]
    
    ss_new_df["hip_flexion_r"] = ss_df["thigh_r"]["angle_x"]
    ss_new_df["hip_rotation_r"] = ss_df["thigh_r"]["angle_y"]
    ss_new_df["hip_adduction_r"] = ss_df["thigh_r"]["angle_z"]
    
    ss_new_df["hip_flexion_l"] = ss_df["thigh_l"]["angle_x"]
    ss_new_df["hip_rotation_l"] = ss_df["thigh_l"]["angle_y"]
    ss_new_df["hip_adduction_l"] = ss_df["thigh_l"]["angle_z"]
    
    ss_new_df["knee_angle_r"] = ss_df["shank_r"]["angle_x"]
    
    ss_new_df["knee_angle_l"] = ss_df["shank_l"]["angle_x"]
    
    ss_new_df["ankle_angle_r"] = ss_df["foot_r"]["angle_x"]
    
    ss_new_df["ankle_angle_l"] = ss_df["foot_l"]["angle_x"]
    
    return ss_new_df
    

# function to plot all the joint angles vs captury angles
def plot_joint_angles(joint_angles, vicon_joint_angles):
    
    fig, axes = plt.subplots(len(list(joint_angles.keys())),1, figsize=(20,12), sharex=True)
       
    for i, joint in enumerate(joint_angles.keys()):
        


        # lim = 180
        # These are time steps to synchronize captury and sensorsuit data
        t_ss = 1359
        t_cap = 0
        
                
        # Plot X axis
        x = np.arange(len(joint_angles[joint]) - t_ss)
        axes[i].plot(x, joint_angles[joint][t_ss:], linewidth=1, color="red")
        
        
        x_vicon = np.arange(len(vicon_joint_angles[joint]))
        axes[i].plot(x_vicon, vicon_joint_angles[joint], linewidth=1, alpha = 0.5, color="red")
        

        # axes[0,i].set_ylim(-lim,lim)
        # axes[1,i].set_ylim(-lim,lim)
        # axes[2,i].set_ylim(-lim,lim)
        # axes[3,i].set_ylim(-lim,lim)
        
        
    fig.suptitle("Joint Angles(Deg)")
    
    plt.tight_layout()
    plt.show()
 
 
def read_mot(vicon_path):
    with open(vicon_path, 'r') as f:
        lines = f.readlines()
        
    # Find the line where actual data starts (usually line starting with 'time')
    for i, line in enumerate(lines):
        if line.strip().startswith('time'):
            data_start = i
            break
        
    # Read the file from the data_start line
    df = pd.read_csv(vicon_path, sep='\t', skiprows=data_start)
    return df
       
# Read the data from vicon and write it to a dataframe
def read_vicon_data(vicon_path):
    
    # Example usage
    mot_path = vicon_path
    df = read_mot(mot_path)

    # for col in df.columns:
    #     print(col)


    Joint_angles_col = ["hip_flexion_r",
                        "hip_rotation_r",
                        "hip_adduction_r",
                        "hip_flexion_l",
                        "hip_rotation_l",
                        "hip_adduction_l",
                        "knee_angle_r",
                        "knee_angle_l",
                        "ankle_angle_r",
                        "subtalar_angle_r",
                        "mtp_angle_r",
                        "ankle_angle_l",
                        "subtalar_angle_l",
                        "mtp_angle_l"]

    angles_df = df[Joint_angles_col]
    
        # angles_df is your original DataFrame with rows sampled at 100 Hz
    original_len = len(angles_df)
    original_index = np.arange(original_len)

    # Desired number of samples for 150 Hz
    new_len = int(np.floor(original_len * 150 / 100))
    new_index = np.linspace(0, original_len - 1, new_len)

    # Use the index as a fake 'x' axis for interpolation
    angles_df.index = original_index

    interpolated_data = {}
    for col in angles_df.columns:
        interpolated_data[col] = np.interp(new_index, original_index, angles_df[col].values)

    # Convert result back into a DataFrame
    angles_df_upsampled = pd.DataFrame(interpolated_data)
    
    print("Vicon data")
    
    angles_df_upsampled = angles_df_upsampled.iloc[0:5700,:]
    
    print(len(angles_df_upsampled))

    return angles_df_upsampled
    
    
    
def main():
    
    vicon_path = "/home/cshah/workspaces/sensorsuit/Vicon Logs/07_09_2025_Nicole/test_SS01_SQ_10_IK.mot"
    csv_path = "/home/cshah/workspaces/sensorsuit/SensorSuit-logs/07_09_2025_Nicole/07_09_2025_GA_Tech_sensor_Config/07_09_SQ_10_01.csv"

    # Load the data to a csv
    data = Utility.load_quaternion_data(csv_path=csv_path)
    
    force = ["L_insole_force", "R_insole_force"]
    insole_force_data = data[force]
    
    
    # Build the Transforms from one frame to the other
    body_transforms = Utility.build_transforms_2()
    limb_structure, joint_heirarchy = set_limb_structure()
    joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole = get_joint_imu_map()
    
    # Extract data
    quaternion_data, t_pose_quat = Utility.extract_data(data=data, 
                                                        joint_imu_map_microstrain=joint_imu_map_microstrain, 
                                                        joint_imu_map_insole=joint_imu_map_insole)
    
    # Extract all acc + gyro data - to add in with quat data for training
    # acc_gyro_data = Utility.extract_acc_gyro_data(all_data=data, joint_imu_map=joint_imu_map)
    # acc_gyro_df = acc_gyro_data[4422:64412]
    
    
    # Transform the quaternions - Zero them to the pelvis frame and transform to animation frame
    transformed_quat, raw_quaternions = transform_quaternions(data=quaternion_data, t_pose_q=t_pose_quat, transforms=body_transforms, joint_heirarchy=joint_heirarchy)
    
    # # Get the joint angles
    joint_angles, joint_rel_raw_quaternions  = cal_joint_angles(transformed_quat, joint_heirarchy, body_transforms, raw_quaternions)
    
    # Read the vicon data
    joint_angles_vicon = read_vicon_data(vicon_path)
    
    # Restreucture joint angles form ss for plotting
    joint_angles_restruct = re_structure_ss_df(joint_angles)
    
    
    

    # # Plot the joint angles
    # plot_joint_angles(joint_angles_restruct, joint_angles_vicon)
    
    # Separated dataframe to write to a csv file
    joint_rel_quat_df = pd.DataFrame([joint_rel_raw_quaternions])
    write_df = pd.DataFrame()
    
    for col in joint_rel_quat_df.columns:
        write_df[f"{col}_qX"] = joint_rel_quat_df[col][0][:,0]
        write_df[f"{col}_qY"] = joint_rel_quat_df[col][0][:,1]
        write_df[f"{col}_qZ"] = joint_rel_quat_df[col][0][:,2]
        write_df[f"{col}_qW"] = joint_rel_quat_df[col][0][:,3]
        
    df_quats_angles = pd.concat([write_df, joint_angles_vicon], axis=1) 
    
    # Write to CSV, formatting floats with up to 6 decimal places
    csv_path = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/GATECH_config/07_09_SQ.csv"
    df_quats_angles.to_csv(csv_path,
          index=False,
          float_format='%.6f',      # e.g. 0.123457
          quoting=csv.QUOTE_MINIMAL)



if __name__ == "__main__":
    raise SystemExit(main())
    