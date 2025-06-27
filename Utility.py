import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import ast
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import csv


## This files provides a range of utility functions to plot all the data from 1 IMU

def plot_imu_data(data_df, key_value_list):

    items = list(key_value_list.items())
  
    for i in range(0, len(items), 2):
        pair = items[i:i+2]  # Get 2 items at a time

        fig1, axes1 = plt.subplots(3,2, figsize=(4,8), sharex=True)
        plot_3_axes_acc(data_df, pair[0][0], pair[0][1], pair[1][0], pair[1][1], axes1)
        fig1.suptitle('IMU Acceleration Data')

        fig2, axes2 = plt.subplots(3,2, figsize=(4,8), sharex=True)
        plot_3_axes_gyro(data_df, pair[0][0], pair[0][1], pair[1][0], pair[1][1], axes2)
        fig2.suptitle('IMU Gyro Data')
    
    
    plt.tight_layout()
    plt.show()


def plot_3_axes_acc(data_df, name1, imu1, name2, imu2, axes): 

    axes[0,0].plot(data_df['Device Time'], data_df[imu1 + '_accel_x'], label = imu1 + '_accel_x')
    axes[0,0].set_title(name1)
    axes[0,0].set_xlabel("Time (s)")
    axes[0,0].set_ylabel("Acceleration (m/s^2)")

    axes[1,0].plot(data_df['Device Time'], data_df[imu1 + '_accel_y'], label = imu1 + '_accel_y')
    # axes[1,num].set_title(name + ' Acceleration Y')
    axes[1,0].set_xlabel("Time (s)")
    axes[1,0].set_ylabel("Acceleration (m/s^2)")

    axes[2,0].plot(data_df['Device Time'], data_df[imu1 + '_accel_z'], label = imu1 + '_accel_z')
    # axes[2,num].set_title(name + ' Acceleration Z')
    axes[2,0].set_xlabel("Time (s)")
    axes[2,0].set_ylabel("Acceleration (m/s^2)")

    axes[0,1].plot(data_df['Device Time'], data_df[imu2 + '_accel_x'], label = imu2 + '_accel_x', color='red')
    axes[0,1].set_title(name2)
    axes[0,1].set_xlabel("Time (s)")
    axes[0,1].set_ylabel("Acceleration (m/s^2)")

    axes[1,1].plot(data_df['Device Time'], data_df[imu2 + '_accel_y'], label = imu2 + '_accel_y', color='red')
    # axes[1,num].set_title(name + ' Acceleration Y')
    axes[1,1].set_xlabel("Time (s)")
    axes[1,1].set_ylabel("Acceleration (m/s^2)")

    axes[2,1].plot(data_df['Device Time'], data_df[imu2 + '_accel_z'], label = imu2 + '_accel_z', color='red')
    # axes[2,num].set_title(name + ' Acceleration Z')
    axes[2,1].set_xlabel("Time (s)")
    axes[2,1].set_ylabel("Acceleration (m/s^2)")


def plot_3_axes_gyro(data_df, name1, imu1, name2, imu2, axes): 

    axes[0,0].plot(data_df['Device Time'], data_df[imu1 + '_gyro_x'], label = imu1 + '_gyro_x')
    axes[0,0].set_title(name1)
    axes[0,0].set_xlabel("Time (s)")
    axes[0,0].set_ylabel("Angular Velocity (rad/s)")

    axes[1,0].plot(data_df['Device Time'], data_df[imu1 + '_gyro_y'], label = imu1 + '_gyro_y')
    # axes[1,num].set_title(name + ' Acceleration Y')
    axes[1,0].set_xlabel("Time (s)")
    axes[1,0].set_ylabel("Angular Velocity (rad/s)")

    axes[2,0].plot(data_df['Device Time'], data_df[imu1 + '_gyro_z'], label = imu1 + '_gyro_z')
    # axes[2,num].set_title(name + ' Acceleration Z')
    axes[2,0].set_xlabel("Time (s)")
    axes[2,0].set_ylabel("Angular Velocity (rad/s)")

    axes[0,1].plot(data_df['Device Time'], data_df[imu2 + '_gyro_x'], label = imu2 + '_gyro_x', color='red')
    axes[0,1].set_title(name2)
    axes[0,1].set_xlabel("Time (s)")
    axes[0,1].set_ylabel("Angular Velocity (rad/s)")

    axes[1,1].plot(data_df['Device Time'], data_df[imu2 + '_gyro_y'], label = imu2 + '_gyro_y', color='red')
    # axes[1,num].set_title(name + ' Acceleration Y')
    axes[1,1].set_xlabel("Time (s)")
    axes[1,1].set_ylabel("Angular Velocity (rad/s)")

    axes[2,1].plot(data_df['Device Time'], data_df[imu2 + '_gyro_z'], label = imu2 + '_gyro_z', color='red')
    # axes[2,num].set_title(name + ' Acceleration Z')
    axes[2,1].set_xlabel("Time (s)")
    axes[2,1].set_ylabel("Angular Velocity (rad/s)")


# Convert a col of strings with multiple entries to float
def string_to_float(col_of_strings):
    return np.array([np.array(ast.literal_eval(q), dtype=np.float64) for q in col_of_strings])


# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    return R.from_quat(q)

#general rotation matrices
def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R



# Calculates basic mean, median, max, min
def calculateStats(arr, name):

    mean = np.mean(arr)
    median = np.median(arr)
    max = np.max(arr)
    min = np.min(arr)

    print(f"{name} mean: {mean}, median: {median}, max: {max}, min: {min}")


# Load quaternion data from CSV
def load_quaternion_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Transforms to go from one frame to the other
def build_transforms():
    transforms = {}

    # # Convert rotation matrices to quaternions
    transforms["Animation_2_pelvis"] = R.from_matrix(get_R_x(np.pi) @ get_R_y( np.pi/2)).as_quat()
    transforms["Animation_2_back"] = R.from_matrix(get_R_x(np.pi) @ get_R_y( np.pi/2)).as_quat()
    # transforms["Animation_2_thigh_r"] = R.from_matrix(get_R_y(-np.pi/2)).as_quat()
    # transforms["Animation_2_thigh_l"] = R.from_matrix(get_R_y(-np.pi/2)).as_quat()
    
    transforms["Animation_2_thigh_l"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(np.pi/2) ).as_quat()
    transforms["Animation_2_thigh_r"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(-np.pi/2) ).as_quat()
    
    # transforms["Animation_2_thigh_l"] = R.from_euler('xyz',[90, -90, 0], degrees=True).as_quat()
    # transforms["Animation_2_thigh_r"] = R.from_euler('xyz',[-90, -90, 0], degrees=True).as_quat()
    
        
    # transforms["Animation_2_shank_l"] = R.from_euler('xyz',[90, -90, 0], degrees=True).as_quat()
    # transforms["Animation_2_shank_r"] = R.from_euler('xyz',[-90, -90, 0], degrees=True).as_quat()
    
    
    transforms["Animation_2_shank_l"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(np.pi/2) ).as_quat()
    transforms["Animation_2_shank_r"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(-np.pi/2) ).as_quat()
    
    ## This s the foot frame when the IMU is on the tongue
    transforms["Animation_2_foot_l"] = R.from_matrix(get_R_z(np.pi)).as_quat()
    transforms["Animation_2_foot_r"] = R.from_matrix(get_R_z(np.pi)).as_quat()
    
    # transforms["Animation_2_foot_l"] = R.from_matrix(np.eye(3)).as_quat()
    # transforms["Animation_2_foot_r"] = R.from_matrix(np.eye(3)).as_quat()
    
    ## Pelvis 2 foot if foot is on the Toe
    transforms["pelvis_2_foot"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    
    ## Pelvis 2 foot if foot is on the Heel
    transforms["pelvis_2_foot_r"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    transforms["pelvis_2_foot_l"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    
    # transforms["pelvis_2_foot_r"] = R.from_matrix(get_R_x(np.pi) @ get_R_y(np.pi/2)).as_quat()
    # transforms["pelvis_2_foot_l"] = R.from_matrix(get_R_x(np.pi) @ get_R_y(np.pi/2)).as_quat()
    
    return transforms

# Transforms to go anatomical frame to sensor frame - all fo these are extrinsic transformations going right to left
def build_transforms_2():
    transforms = {}
    
    # Anatomical 2 Sensor Frames
    transforms["Anatomical_2_pelvis"] = R.from_matrix( get_R_z(np.pi) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_back"] = R.from_matrix( get_R_z(np.pi) @ get_R_y(-np.pi/2) )
    
    transforms["Anatomical_2_thigh_r"] = R.from_matrix( get_R_z(-np.pi/2) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_thigh_l"] = R.from_matrix( get_R_z(np.pi/2) @ get_R_y(-np.pi/2) )
    
    transforms["Anatomical_2_shank_r"] = R.from_matrix( get_R_z(-np.pi/2) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_shank_l"] = R.from_matrix( get_R_z(np.pi/2) @ get_R_y(-np.pi/2) )
    
    transforms["Anatomical_2_foot_r"] = R.from_matrix( get_R_z(np.pi/2) @ get_R_y(-np.pi/2))
    transforms["Anatomical_2_foot_l"] = R.from_matrix( get_R_z(-np.pi/2) @ get_R_y(-np.pi/2))
    
    
    # Pelvis 2 Sensor Frames
    
    transforms["pelvis_2_pelvis"] = R.from_matrix( np.eye(3)) 
    
    transforms["pelvis_2_thigh_r"] = R.from_matrix( get_R_x(np.pi/2) )
    transforms["pelvis_2_thigh_l"] = R.from_matrix( get_R_x(-np.pi/2) )
    
    transforms["pelvis_2_shank_r"] = R.from_matrix( get_R_x(np.pi/2) )
    transforms["pelvis_2_shank_l"] = R.from_matrix( get_R_x(-np.pi/2) )
    
    # transforms["pelvis_2_foot"] = R.from_matrix( get_R_x(np.pi) @ get_R_y(np.pi/2) )
    transforms["pelvis_2_foot"] = R.from_matrix( get_R_y(np.pi/2) )
    
    transforms["pelvis_2_foot_r"] = R.from_matrix( get_R_x(-np.pi/2) )
    transforms["pelvis_2_foot_l"] = R.from_matrix( get_R_x(np.pi/2) )
    
    return transforms

def average_quaternions(quats):
    """Averages a list of quaternions using SLERP-based incremental averaging."""
    if len(quats) == 0:
        raise ValueError("No quaternions to average.")

    # Start with the first quaternion as the base
    avg_q = R.from_quat(quats[0])
    for i in range(1, len(quats)):
        next_q = R.from_quat(quats[i])
        # Slerp between current average and next quaternion
        slerp = Slerp([0, 1], R.from_quat([avg_q.as_quat(), next_q.as_quat()]))
        avg_q = slerp(0.5)  # halfway SLERP
    return avg_q

# Function to extract the imu and insole data and return a list with the entire data and a tpose list
def extract_data(data, joint_imu_map_microstrain, joint_imu_map_insole):
    
    t_pose_q = {limb : np.array(eval(data[f"{imu}"].iloc[0])) for limb, imu in joint_imu_map_microstrain.items()}

    
    # # Extract t_pose_q for microstrain
    # t_pose_q = {}

    # for limb, imu in joint_imu_map_microstrain.items():
    #     # Read the first 50 quaternion frames for this IMU
    #     quat_array = np.stack([np.array(eval(row)) for row in data[imu].iloc[0]])
    
    #     # Normalize quaternions (in case they're not unit norm)
    #     quat_array /= np.linalg.norm(quat_array, axis=1, keepdims=True)
    
    #     # Average the quaternions using SLERP-based approach
    #     t_pose_q[limb] = average_quaternions(quat_array)
    
    
    # Extract quat for microstrain
    quat_data = {limb: np.stack(data[f"{imu}"].apply(eval).values) for limb, imu in joint_imu_map_microstrain.items()}
    
    
    # Extract Insole quaternion data
    for limb, data_key in joint_imu_map_insole.items():
        
        cols = [data_key+"_qw", data_key+"_qx" , data_key+"_qy" , data_key+"_qz" ]
        
        quat_cols = data[cols]
        quat_data[limb] = quat_cols.to_numpy()
        
        t_pose_q[limb] = quat_cols.to_numpy()[0]

        
        # # Convert the first 50 string quaternions to np.array
        # quat_array_foot = quat_cols.to_numpy()[:50]

        # # Normalize the quaternions
        # quat_array_foot /= np.linalg.norm(quat_array_foot, axis=1, keepdims=True)

        # # Average them using SLERP or your preferred method
        # t_pose_q[limb] = average_quaternions(quat_array_foot)

        
    return quat_data, t_pose_q
        
# Combine the sensorsuit data frame and the captury dataframe and write the whole dataframe as a csv so we can use it for training
def write_dataframe_2_csv(dict_1, dict_2, acc_gyro_df):
    
    df_1 = pd.DataFrame([dict_1])
    df_2 = pd.DataFrame([dict_2])
    
    print(df_1.shape)
    print(df_2.shape)
    
    # The general assumption when writing this function is that the dataframes are the same length
    df_combined = pd.concat([df_1, df_2], axis=1)
    
    # Separated dataframe to write to a csv file
    sep_df = pd.DataFrame()
    
    for col in df_combined.columns:
        sep_df[f"{col}_qX"] = df_combined[col][0][:,0]
        sep_df[f"{col}_qY"] = df_combined[col][0][:,1]
        sep_df[f"{col}_qZ"] = df_combined[col][0][:,2]
        sep_df[f"{col}_qW"] = df_combined[col][0][:,3]
        
    # The general assumption when writing this function is that the dataframes are the same length
    df_quats_acc_gyro = pd.concat([sep_df, acc_gyro_df], axis=1) 
    
    

    # Write to CSV, formatting floats with up to 6 decimal places
    csv_path = "/home/cshah/workspaces/deepPhase based work/Data/Quaternion_training_data_pelvis_frame_rel_quats_with_acc_gyro.csv"
    df_quats_acc_gyro.to_csv(csv_path,
          index=False,
          float_format='%.6f',      # e.g. 0.123457
          quoting=csv.QUOTE_MINIMAL)


# This function gets the entire data frame extracts the required data and returns it as dataframe subset
def extract_acc_gyro_data(all_data, joint_imu_map):
    
    columns_to_extract = []
    
    for keys, cols in joint_imu_map.items():
        
        
        columns_to_extract.append(cols+"_accel_x")
        columns_to_extract.append(cols+"_accel_y")
        columns_to_extract.append(cols+"_accel_z")
        
        
    for keys, cols in joint_imu_map.items():
        columns_to_extract.append(cols+"_gyro_x")
        columns_to_extract.append(cols+"_gyro_y")
        columns_to_extract.append(cols+"_gyro_z")
        
    
        
      
    # Extract all the Gyro's
    subset_df = all_data[columns_to_extract].copy()    
        
    return subset_df