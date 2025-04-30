################ Author - Chinmay Shah ##########################
'''
The goal of the file is to plot all the data.
'''


import os
import pandas as pd
import re
import Utility 
import matplotlib.pyplot as plt
import numpy as np


# This function gets the entire data frame extracts the required data and returns it as dataframe subset
def extract_gyro_data(all_data, data_keys):
    
    columns_to_extract = []
    
    for keys, cols in data_keys.items():
        
        columns_to_extract.append(cols+"_gyro_x")
        columns_to_extract.append(cols+"_gyro_y")
        columns_to_extract.append(cols+"_gyro_z")
        
        
    subset_df = all_data[columns_to_extract].copy()
        
    return subset_df


def plot_all_gyro_data(df, file_name, col_names, data_keys):
    
    fig, axs = plt.subplots(3, 8, figsize=(30,10), sharex=True)
    
    key_list = list(data_keys.keys())
            
    for i in range(24):
        row = i % 3
        col = i // 3
        ax = axs[row, col]
            
        ax.plot(df.iloc[:, i], linewidth=2, color="black")  # Plot the i-th column
                
        joint_name = None
        prefix = col_names[i][:4]
                
        for k in key_list:
                    
            if prefix in k:
                joint_name = data_keys[k]
                break
                    
        if "_l" in joint_name:
            joint_name = joint_name.replace("_l", "")
            joint_name = "left " + joint_name
        elif "_r" in joint_name:
            joint_name = joint_name.replace("_r", "")
            joint_name = "right " + joint_name
                
        name = joint_name + " (" + col_names[i] + ")"
        ax.set_title(name)
        ax.tick_params(labelsize=8)
                
    
    fig.suptitle(file_name)

    plt.tight_layout()
    plt.show()
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    

# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_21_2025/04_21_2025_leg_swing.csv"
csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_28_2025/04_28_2025_tap_test.csv"


df = pd.read_csv(csv_path)

data_keys = {
    "back": "imu3",
    "pelvis": "imu2",
    "thigh_l": "imu1",
    "thigh_r": "imu6", 
    "shank_l": "imu4",
    "shank_r": "imu5",
    "foot_l": "L_insole",
    "foot_r": "R_insole"
}

col_keys = {
    "imu3": "back",
    "imu2": "pelvis",
    "imu1": "thigh_l",
    "imu6": "thigh_r", 
    "imu4": "shank_l",
    "imu5": "shank_r",
    "L_insole": "foot_l",
    "R_insole": "foot_r"
}

extracted_df = extract_gyro_data(df, data_keys)

col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
                 "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z"]


for col in col_to_modify:
    extracted_df[col] = extracted_df[col] * np.pi / 180
    

plot_all_gyro_data(extracted_df, "All_gyro_data.png", extracted_df.columns, col_keys)
    
    
