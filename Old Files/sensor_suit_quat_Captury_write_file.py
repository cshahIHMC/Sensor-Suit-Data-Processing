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
        "pelvis": "imu2",
        "thigh_r": "imu5", 
        "shank_r": "imu4",
        "thigh_l": "imu1",
        "shank_l": "imu6",
        # "back": "imu3",
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
def transform_quaternions(data, t_pose_q, transforms, joint_heirarchy, t_plot, t_ss_end):
    
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
             
        raw_quaternions[f"{imu}_ss"] = (relative_rotations[t_plot:t_ss_end])
        
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

# function to plot all the joint angles vs captury angles
def plot_joint_angles(joint_angles, GRF, captury_joint_angles, t_plot):
    
    fig, axes = plt.subplots(4, len(list(joint_angles.keys())), figsize=(20,12), sharex=True)
       
    for i, joint in enumerate(joint_angles.keys()):
        
        joint_angle = joint_angles[joint]
        captury_joint_angle = captury_joint_angles[joint]
        
        # lim = 180
        # These are time steps to synchronize captury and sensorsuit data
        t_ss = t_plot
        t_cap = 0
        
        if "_r" in joint:
            factor = -1
        else:
            factor = 1
        
        
        # Plot X axis
        axes[0,i].plot(joint_angle["angle_x"][t_ss:], linewidth=1, color="red")
        axes[0,i].plot(factor * captury_joint_angle[t_cap:,0], linewidth=1, alpha = 0.5, color="red")
        axes[0,i].set_title(joint + "_X (deg)")
        
        # Plot Y axis
        axes[1,i].plot(joint_angle["angle_y"][t_ss:], linewidth=1, color="green")
        axes[1,i].plot(-factor * captury_joint_angle[t_cap:,1], linewidth=1, alpha = 0.5, color="green")
        axes[1,i].set_title(joint + "_Y (deg)")
    
        # Plot Z axis
        axes[2,i].plot(joint_angle["angle_z"][t_ss:], linewidth=1, color="blue")
        axes[2,i].plot(captury_joint_angle[t_cap:,2], linewidth=1, alpha = 0.5, color="blue")
        axes[2,i].set_title(joint + "_Z (deg)")
        
        if "_r" in joint:
            axes[3,i].plot(GRF["R_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Right GRF")
        else:
            axes[3,i].plot(GRF["L_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Left GRF")
            
        # axes[0,i].set_ylim(-lim,lim)
        # axes[1,i].set_ylim(-lim,lim)
        # axes[2,i].set_ylim(-lim,lim)
        # axes[3,i].set_ylim(-lim,lim)
        
        
    fig.suptitle("Joint Angles(Deg)")
    
    plt.tight_layout()
    plt.show()

## Function to read all the data from captury
def read_captury_data(path, original_end, total_data_end):
    csv_path = path


    # If you need to specify an encoding or handle bad lines:
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=2,
        encoding="utf-8",       # e.g. latin1, utf-8, etc.
        engine="python",        # sometimes needed for complex separators
    )

    # Extract the required quaternions
    # joint → tuple(start_idx, start_idx+1, start_idx+2, start_idx+3)
    joint_idxs = {
        "foot_l":  (58, 59, 60, 61),
        "shank_l": (65, 66, 67, 68),
        "thigh_l": (72, 73, 74, 75),
        "foot_r":  (86, 87, 88, 89),
        "shank_r": (93, 94, 95, 96),
        "thigh_r": (100,101,102,103),
        "pelvis":  (177,178,179,180),
    }
    comps = ["X","Y","Z","W"]


    # Create an empty DataFrame
    quat_df = pd.DataFrame()
    
    t_end = original_end

    # Loop over each joint, pull out its 4 columns, and assign them with nice names
    for joint, (i0,i1,i2,i3) in joint_idxs.items():
        # grab the raw (n_rows × 4) slice
        mat = df.iloc[2:t_end, [i0, i1, i2, i3]].to_numpy()  
        mat = mat.astype(np.float64)
        # pack each row into one object: either a list or small ndarray
        quat_df[joint] = [row.copy() for row in mat]
        
    print(quat_df.shape)

    # 4) Now loop over columns and convert each series into a SciPy Rotation
    rotations = {}
    for col in quat_df.columns:
        # stack into shape (n_frames,4)
        all_quats = np.stack(quat_df[col].values, axis=0)  
        # create one Rotation object per frame
        # 1) compute norms
        norms = np.linalg.norm(all_quats, axis=1)

        # 2) find zeros
        zero_mask = norms == 0

        # 3) replace zero‐rows with identity quaternion
        all_quats[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0])

        # 4) normalize _all_ quaternions (including the ones you just fixed)
        all_quats = all_quats / np.linalg.norm(all_quats, axis=1, keepdims=True)

        # 5) now safely convert
        rotations[col] = R.from_quat(all_quats)
        

    joint_heirarchy = {
        "pelvis" : "pelvis",
        "thigh_r" : "pelvis",
        "thigh_l" : "pelvis",
        "shank_r" : "thigh_r",
        "shank_l" : "thigh_l",
        "foot_r" : "shank_r",
        "foot_l" : "shank_l"        
    }


    joint_angles = {}
    relative_joint_quaternions = {}
    relative_joint_quaternions_upsampled = {}
    joint_angles_upsampled = {}
    
    # from captury we directly get rel child limb quaternions
    for child, parent in joint_heirarchy.items():
        
        relative_joint_quaternions[child] = rotations[child]

        # original rotations (Rotation object of length N)
        rots = relative_joint_quaternions[child]

        # original timestamps (e.g. frame numbers or seconds)
        N_orig = len(rots)
        # 1) sampling rates
        fs_orig = 70.0    # original Hz
        fs_new  = 150.0   # desired Hz

        # 2) total duration in seconds
        duration = (N_orig - 1) / fs_orig

        # 3) create time vectors
        t_orig = np.linspace(0.0, duration, N_orig)                  # shape (N_orig,)
        N_new  = int(np.round(duration * fs_new)) + 1                # number of new samples

        # new timestamps, 3× as many
        t_new = np.linspace(0, duration, N_new)

        # build the slerp interpolator
        slerp = Slerp(t_orig, rots)

        # evaluate at high‐rate times
        high_rate_rots = slerp(t_new)        # Rotation of length N*3
        
        t_cap_write_start = 0
        t_cap_write_end = total_data_end
        relative_joint_quaternions_upsampled[f"{child}_captury"] = high_rate_rots[t_cap_write_start:t_cap_write_end].as_quat()

        # now extract your Euler angles
        joint_angles_upsampled[child] = high_rate_rots.as_euler('zyx', degrees=True)
        
        print(len(relative_joint_quaternions_upsampled[f"{child}_captury"] ))
        # joint_angles_upsampled[child] = high_rate_rots.as_quat()
        
        
    return joint_angles_upsampled, relative_joint_quaternions_upsampled
    

def main():
    
    captury_path = "/home/cshah/workspaces/sensorsuit/Captury logs/07_09_2025_Nicole/Chinmay Sensor Config/07_09_Nicole_random.csv"
    csv_path = "/home/cshah/workspaces/sensorsuit/SensorSuit-logs/07_09_2025_Nicole/07_09_2025_Chinmay_Sensor_Config/07_09_C_random.csv"

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
    

    
    
    cap_original_end = 8200
    cap_total_data_end = 17500
    
    
    t_plot = 905
    t_ss_end = 18405
    
    
    
    # Extract all acc + gyro data - to add in with quat data for training
    acc_gyro_data = Utility.extract_acc_gyro_data(all_data=data, joint_imu_map=joint_imu_map)
    acc_gyro_df = acc_gyro_data[t_plot:t_ss_end]
    
    # Transform the quaternions - Zero them to the pelvis frame and transform to animation frame
    transformed_quat, raw_quaternions = transform_quaternions(data=quaternion_data, t_pose_q=t_pose_quat, transforms=body_transforms, joint_heirarchy=joint_heirarchy, t_plot=t_plot, t_ss_end=t_ss_end)
    
    # # Get the joint angles
    joint_angles, joint_rel_raw_quaternions  = cal_joint_angles(transformed_quat, joint_heirarchy, body_transforms, raw_quaternions)
    
    # Extract the Captury Data
    captury_joint_angles, captury_joint_quaternions = read_captury_data(captury_path, cap_original_end, cap_total_data_end)
    

    # # Plot the joint angles
    plot_joint_angles(joint_angles, insole_force_data, captury_joint_angles, t_plot)
    
    # Write all the data into one file for ML training
    # Utility.write_dataframe_2_csv(joint_rel_raw_quaternions, captury_joint_quaternions)
    # # Write to CSV, formatting floats with up to 6 decimal places
    # csv_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/07_09_RM_gyro.csv"
    # acc_gyro_df.to_csv(csv_path,
    #       index=False,
    #       float_format='%.6f',      # e.g. 0.123457
    #       quoting=csv.QUOTE_MINIMAL)




if __name__ == "__main__":
    raise SystemExit(main())
    