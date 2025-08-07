import pandas as pd
import Utility
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import csv


def read_data(vicon_data_path, sensor_suit_data_path):
    
    ss_df = pd.read_csv(sensor_suit_data_path) 
    vicon_df = Utility.read_mot(vicon_data_path)
    
    return vicon_df, ss_df

def extract_vicon_data(vicon_df):
    
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
    
    angles_df = vicon_df[Joint_angles_col]
    
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
    
    print("Vicon Data Length")
    print(len(angles_df_upsampled))
    
    print("Vicon Data start point")
    data_start_point = vicon_df["time"].iloc[0]
    print(data_start_point)
    data_length = len(angles_df_upsampled)
    
    print("Vicon Data Shape")
    print(angles_df_upsampled.shape)
    
    # # Mirror the vicon data 
    # angles_df_upsampled_mirror = pd.DataFrame()
    # angles_df_upsampled_mirror["hip_flexion_r"] = angles_df_upsampled["hip_flexion_l"]
    # angles_df_upsampled_mirror["hip_rotation_r"] = angles_df_upsampled["hip_rotation_l"]
    # angles_df_upsampled_mirror["hip_adduction_r"] = angles_df_upsampled["hip_adduction_l"]
    # angles_df_upsampled_mirror["hip_flexion_l"] = angles_df_upsampled["hip_flexion_r"]
    # angles_df_upsampled_mirror["hip_rotation_l"] = angles_df_upsampled["hip_rotation_r"]
    # angles_df_upsampled_mirror["hip_adduction_l"] = angles_df_upsampled["hip_adduction_r"]
    # angles_df_upsampled_mirror["knee_angle_r"] = angles_df_upsampled["knee_angle_l"]
    # angles_df_upsampled_mirror["knee_angle_l"] = angles_df_upsampled["knee_angle_r"]
    # angles_df_upsampled_mirror["ankle_angle_r"] = angles_df_upsampled["ankle_angle_l"]
    # angles_df_upsampled_mirror["ankle_angle_l"] = angles_df_upsampled["ankle_angle_r"]
    
    # print("Vicon Data Mirrored")
    # print(angles_df_upsampled_mirror.shape)
    
    # angles_df_combined = pd.concat([angles_df_upsampled, angles_df_upsampled_mirror], axis=0)

    # print("Vicon Data All")
    # print(angles_df_combined.shape)

    return angles_df_upsampled, data_start_point, data_length

def extract_acc_gyro_insole(sensor_suit_df, sync_start_point, sync_stop_point, joint_imu_map):
    
    columns_to_extract = []
    
    acc_gyro_insole_df = pd.DataFrame()
    
    # Gyro
    for keys, cols in joint_imu_map.items():
        
        acc_gyro_insole_df[keys+"_gyro_x"] = sensor_suit_df[cols+"_gyro_x"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_gyro_y"] = sensor_suit_df[cols+"_gyro_y"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_gyro_z"] = sensor_suit_df[cols+"_gyro_z"].iloc[sync_start_point:sync_stop_point]   
        
        # The xsensor data is recorded in deg/sec while microstrain data is recorded in rad/sec.
        # To make everything consistent we convert the xsensor data to rad/sec
        if "insole" in cols:
            acc_gyro_insole_df[keys+"_gyro_x"] = acc_gyro_insole_df[keys+"_gyro_x"] * np.pi / 180
            acc_gyro_insole_df[keys+"_gyro_y"] = acc_gyro_insole_df[keys+"_gyro_y"] * np.pi / 180
            acc_gyro_insole_df[keys+"_gyro_z"] = acc_gyro_insole_df[keys+"_gyro_z"] * np.pi / 180
            
     
    
    # Acc    
    for keys, cols in joint_imu_map.items():   
        acc_gyro_insole_df[keys+"_accel_x"] = sensor_suit_df[cols+"_accel_x"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_accel_y"] = sensor_suit_df[cols+"_accel_y"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_accel_z"] = sensor_suit_df[cols+"_accel_z"].iloc[sync_start_point:sync_stop_point]   
      
    # Insole
    insole_map = {
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    for keys, cols in insole_map.items():   
        acc_gyro_insole_df[keys+"_force"] = sensor_suit_df[cols+"_force"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_COPx"] = sensor_suit_df[cols+"_COPx"].iloc[sync_start_point:sync_stop_point]
        acc_gyro_insole_df[keys+"_COPz"] = sensor_suit_df[cols+"_COPz"].iloc[sync_start_point:sync_stop_point]
    
    print("Gyro Acc Insole df shape")
    print(acc_gyro_insole_df.shape)
    return acc_gyro_insole_df

# Apply transformations to the quaternions
# - Zero with Pelvis
# - Bring them to animation frame
def transform_quaternions(data, t_pose_q, joint_map):
    
    transformed_data = {}
    quat_norm = {}
    
    # Get the t pose quat and normalize them
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}

    # Loop over all IMU's and according to their frames apply the correct transformations
    for joint, _ in joint_map.items():
            
        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[joint], axis=1, keepdims=True)
        quat_norm[joint] = R.from_quat(data[joint]/quat_magnitude, scalar_first=True)

        if "foot_r" in joint:
            # 10min in Calibration
            transform_r = [ 0.96805752,  0.24972168, -0.01774981,  0.01163004]
            # relative_rotations =  (R.from_quat(transform_r) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_r) * quat_norm[imu] 
            A = R.from_quat(transform_r) * quat_norm[joint]
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_r) * t_pose_q_norm[joint]
            D = t_pose_q_norm["pelvis"].inv() * C
            relative_rotations = D.inv() * B 
            
        elif "foot_l" in joint:

            # 10min in Calibration
            transform_l = [ 0.21206732,  0.97710702, -0.01205927, -0.00935059]
 
            # relative_rotations =      (R.from_quat(transform_l) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_l) * quat_norm[imu] 
            A = R.from_quat(transform_l) * quat_norm[joint]
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_l) * t_pose_q_norm[joint]
            D = t_pose_q_norm["pelvis"].inv() * C
            relative_rotations =  D.inv() * B 
        else:
            # relative_rotations = t_pose_q_norm[imu].inv() * quat_norm[imu] 
            A = quat_norm["pelvis"].inv() * quat_norm[joint]
            B = t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[joint]
            
            relative_rotations = B.inv() * A 
        
        # Stored the transformed_data
        transformed_data[joint+"_quat"] = relative_rotations
        
    return transformed_data

# Calculate the joint angles using the transformed quaternion data
def cal_joint_angles(quaternion_data, joint_heirarchy):
    joint_quaternions = {}
    joint_angles = {}
    
    
    # Iterate over all imu's in joint heirarchy
    for child, parent in joint_heirarchy.items():
        
        child_joint = child + "_quat"
        parent_joint = parent + "_quat"
        
        if "pelvis" in child or "thigh" in child:
            joint_quaternions[child_joint] = quaternion_data[child_joint]
        else:
            joint_quaternions[child_joint] = quaternion_data[parent_joint].inv() * quaternion_data[child_joint]
            
        joint_angles[child] = joint_quaternions[child_joint].as_euler("zyx", degrees = True)
        
        joint_quaternions[child_joint] = joint_quaternions[child_joint].as_quat()
        # Normalzing the Quaternions that are sent from the sensor suit side   
        joint_quaternions[child_joint] = joint_quaternions[child_joint] / np.linalg.norm(joint_quaternions[child_joint] , axis=1, keepdims=True) 
        
        
         
    return joint_quaternions, joint_angles


def extract_quat_df(sensor_suit_df, sync_start_point, sync_stop_point, joint_imu_map, joint_insole_map, joint_map, joint_heirarchy):
    
    t_pose_q = {limb : np.array(eval(sensor_suit_df[f"{imu}_quat"].iloc[0])) for limb, imu in joint_imu_map.items()}
    quat_data = {limb: np.stack(sensor_suit_df[f"{imu}_quat"].iloc[sync_start_point:sync_stop_point].apply(eval).values) for limb, imu in joint_imu_map.items()}
    
    # Extract Insole quaternion data
    for limb, data_key in joint_insole_map.items():
        
        cols = [data_key+"_qw", data_key+"_qx" , data_key+"_qy" , data_key+"_qz" ]
        
        quat_cols = sensor_suit_df[cols]
        quat_data[limb] = quat_cols.iloc[sync_start_point:sync_stop_point].to_numpy()
        
        t_pose_q[limb] = quat_cols.to_numpy()[0]
        
    # Transform Quat to pelvis frame and zero to the first frame
    transformed_quat = transform_quaternions(quat_data, t_pose_q, joint_map)
    
    # Calculate the joint rel quaternions and joint angles
    joint_rel_quaternions, joint_angles = cal_joint_angles(transformed_quat, joint_heirarchy)
    
    # Convert this to a dataframe
    joint_quat_df = pd.DataFrame([joint_rel_quaternions])
    quat_df = pd.DataFrame()

    for col in joint_quat_df.columns:
        quat_df[f"{col}_qX"] = joint_quat_df[col][0][:,0]
        quat_df[f"{col}_qY"] = joint_quat_df[col][0][:,1]
        quat_df[f"{col}_qZ"] = joint_quat_df[col][0][:,2]
        quat_df[f"{col}_qW"] = joint_quat_df[col][0][:,3]
        
    return quat_df, joint_angles
         

    
    
    
    
def extract_ss_data(sensor_suit_df, data_start_point, data_length, joint_map, joint_imu_map, joint_insole_map, subject_parameters, joint_heirarchy):
    
    print("Sensor Suit Raw Data Shape")
    print(sensor_suit_df.shape)
    # Using the sync line find the data start point
    # its the first value 0 becomes 1 in the SYNC line
    # We add it with 150 * data_start_point if the vicon data was cropped
    sync_start_index = sensor_suit_df[(sensor_suit_df['SYNC'].shift(1) == 0) & (sensor_suit_df['SYNC'] == 1)].index[0] 
    
    sync_start_point = round(sync_start_index + data_start_point * 150)
    sync_stop_point = sync_start_point + data_length

    gyro_acc_insole_df = extract_acc_gyro_insole(sensor_suit_df, sync_start_point, sync_stop_point, joint_map)
    
    # Subject parameters df    
    num_rows = len(gyro_acc_insole_df)
    subject_parameters_df = pd.DataFrame({k: [v] * num_rows for k, v in subject_parameters.items()})
    
    print("Subject Parameters df shape")
    print(subject_parameters_df.shape)
    
    # Extract the quaternions in the pelvis frame
    quat_df, joint_angles  = extract_quat_df(sensor_suit_df, sync_start_point, sync_stop_point, joint_imu_map, joint_insole_map, joint_map, joint_heirarchy)
    
    print("Quat df shape")
    print(quat_df.shape)
    
    
    return gyro_acc_insole_df, subject_parameters_df, quat_df
    
    
    


    
def main():
    
    vicon_data = "/home/cshah/workspaces/sensorsuit/Vicon Logs/07_16_2025/Walk_Around_results.mot"
    sensor_suit_data = "/home/cshah/workspaces/sensorsuit/SensorSuit-logs/07_16_2025_Chinmay/Chinmay Sensor Config/07_16_2_walk_around.csv"
    
    # Extract the pandas data frame
    vicon_df, sensor_suit_df = read_data(vicon_data, sensor_suit_data)

    # Extract the required data
    vicon_df, data_start_point, data_length = extract_vicon_data(vicon_df)
    
    # Extract the ss data
    
    joint_map = {
        "pelvis" : "imu2",
        "thigh_r" : "imu1",
        "thigh_l" : "imu6",
        "shank_r" : "imu4",
        "shank_l" : "imu5",
        "foot_r" : "R_insole",
        "foot_l" : "L_insole"
    }
    
    joint_imu_map = {
        "pelvis" : "imu2",
        "thigh_r" : "imu1",
        "thigh_l" : "imu6",
        "shank_r" : "imu4",
        "shank_l" : "imu5"
    }
    
    joint_insole_map = {
        "foot_r" : "R_insole",
        "foot_l" : "L_insole"
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
    
    subject_parameters = {
        "Age" : 28,
        "Sex" : 0, # 0 for male, 1 for female
        "Height": 172, # in cm
        "Weight": 82, # in kgs
        "Pelvis_width" : "",  # in cm
        "Thigh_r_length": "",
        "Thigh_l_length": "",
        "Shank_r_length": "",
        "Shank_l_length": "",
        "Foot_r_length": "",
        "Foot_l_length": ""    
    }
    
    gyro_acc_insole_df, subject_paramters_df, quat_df = extract_ss_data(sensor_suit_df, data_start_point, data_length, joint_map, joint_imu_map, joint_insole_map, subject_parameters, joint_heirarchy)
    
    quat_2_angles_df = pd.concat([quat_df, vicon_df], axis=1)
    
    # Write to CSV, formatting floats with up to 6 decimal places
    csv_path = "/home/cshah/workspaces/deepPhase based work/Data/Vicon_SS_Data_trial/EST_sub01_walk_around.csv"
    quat_2_angles_df.to_csv(csv_path,
          index=False,
          float_format='%.6f',      # e.g. 0.123457
          quoting=csv.QUOTE_MINIMAL)    
    
    
if __name__ == "__main__":
    raise SystemExit(main())
    