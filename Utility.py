import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import ast
import numpy as np
from scipy.spatial.transform import Rotation as R


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
