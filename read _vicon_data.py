import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

def read_mot(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the line where actual data starts (usually line starting with 'time')
    for i, line in enumerate(lines):
        if line.strip().startswith('time'):
            data_start = i
            break

    # Read the file from the data_start line
    df = pd.read_csv(filename, sep='\t', skiprows=data_start)
    return df

# Example usage
mot_path = "/home/cshah/workspaces/sensorsuit/Vicon Logs/07_16_2025/FJC_results.mot"
df = read_mot(mot_path)

for col in df.columns:
    print(col)


# Joint_angles_col = ["hip_flexion_r",
#                     "hip_rotation_r",
#                     "hip_adduction_r",
#                     "hip_flexion_l",
#                     "hip_rotation_l",
#                     "hip_adduction_l",
#                     "knee_angle_r",
#                     "knee_angle_l",
#                     "ankle_angle_r",
#                     "subtalar_angle_r",
#                     "mtp_angle_r",
#                     "ankle_angle_l",
#                     "subtalar_angle_l",
#                     "mtp_angle_l"]

# angles_df = df[Joint_angles_col]

# print(len(angles_df))



# # Read captury data
# csv_path = "/home/cshah/workspaces/sensorsuit/Captury logs/07_09_2025_Nicole/GAtech sensor config/07_09_SS_Nicole_TW_08.csv"
# # If you need to specify an encoding or handle bad lines:
# df = pd.read_csv(
#     csv_path,
#     sep=";",
#     header=2,
#     encoding="utf-8",       # e.g. latin1, utf-8, etc.
#     engine="python",        # sometimes needed for complex separators
# )
# # Hard coded col numbers of information to input
# # Extract the required quaternions
# # joint → tuple(start_idx, start_idx+1, start_idx+2, start_idx+3)
# joint_idxs = {
#     "foot_l":  (58, 59, 60, 61),
#     "shank_l": (65, 66, 67, 68),
#     "thigh_l": (72, 73, 74, 75),
#     "foot_r":  (86, 87, 88, 89),
#     "shank_r": (93, 94, 95, 96),
#     "thigh_r": (100,101,102,103),
#     "pelvis":  (177,178,179,180),
# }
# comps = ["X","Y","Z","W"]
# # Create an empty DataFrame
# quat_df = pd.DataFrame()
# # Loop over each joint, pull out its 4 columns, and assign them with nice names
# # Rows number from till where to extract data
# t1 = 2
# t2 = 7370
# for joint, (i0,i1,i2,i3) in joint_idxs.items():
#     # grab the raw (n_rows × 4) slice
#     mat = df.iloc[t1:t2, [i0, i1, i2, i3]].to_numpy()  
#     mat = mat.astype(np.float64)
#     # pack each row into one object: either a list or small ndarray
#     quat_df[joint] = [row.copy() for row in mat]
# print(len(quat_df))
# # 4) Now loop over columns and convert each series into a SciPy Rotation
# rotations = {}
# for col in quat_df.columns:
#     # stack into shape (n_frames,4)
#     all_quats = np.stack(quat_df[col].values, axis=0)  
#     # create one Rotation object per frame
#     # 1) compute norms
#     norms = np.linalg.norm(all_quats, axis=1)
#     # 2) find zeros
#     zero_mask = norms == 0
#     # 3) replace zero‐rows with identity quaternion
#     all_quats[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0])
#     # 4) normalize _all_ quaternions (including the ones you just fixed)
#     all_quats = all_quats / np.linalg.norm(all_quats, axis=1, keepdims=True)
#     # 5) now safely convert
#     rotations[col] = R.from_quat(all_quats)
# joint_heirarchy = {
#     "pelvis" : "pelvis",
#     "thigh_r" : "pelvis",
#     "thigh_l" : "pelvis",
#     "shank_r" : "thigh_r",
#     "shank_l" : "thigh_l",
#     "foot_r" : "shank_r",
#     "foot_l" : "shank_l"        
# }
# # Convert quaternions to joint angles
# # The quaternions reported by captury are already in the locla segment frame
# joint_angles = {}
# relative_joint_quaternions = {}
# for child, parent in joint_heirarchy.items():
#     relative_joint_quaternions[child] = rotations[child]
    
#     # original rotations (Rotation object of length N)
#     rots = relative_joint_quaternions[child]
#     # original timestamps (e.g. frame numbers or seconds)
#     N_orig = len(rots)
#     # 1) sampling rates
#     fs_orig = 70.0    # original Hz
#     fs_new  = 100.0   # desired Hz
#     # 2) total duration in seconds
#     duration = (N_orig - 1) / fs_orig
#     # 3) create time vectors
#     t_orig = np.linspace(0.0, duration, N_orig)                  # shape (N_orig,)
#     N_new  = int(np.round(duration * fs_new)) + 1                # number of new samples
#     # new timestamps, 3× as many
#     t_new = np.linspace(0, duration, N_new)
#     # build the slerp interpolator
#     slerp = Slerp(t_orig, rots)
#     # evaluate at high‐rate times
#     high_rate_rots = slerp(t_new)

#     joint_angles[child] = high_rate_rots.as_euler('xyz', degrees=True)


# fig, axes = plt.subplots(3, 6, figsize=(20,12), sharex=True)

# t1 = 377
# # Hip Angle Right    
# # Plot X axis
# axes[0,0].plot(joint_angles["thigh_r"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,0].plot(angles_df["hip_flexion_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,0].set_title( "Right Hip_X (deg)")
# axes[0,0].legend()
        
# # Plot Y axis
# axes[1,0].plot(joint_angles["thigh_r"][t1:,1], linestyle='-', linewidth=0.7, color="green", label="Captury")
# axes[1,0].plot(angles_df["hip_rotation_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[1,0].set_title("Right Hip_Y (deg)")
# axes[1,0].legend()

# # Plot Z axis
# axes[2,0].plot(joint_angles["thigh_r"][t1:,2], linestyle='-', linewidth=0.7, color="blue", label="Captury")
# axes[2,0].plot(-angles_df["hip_adduction_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[2,0].set_title("Right Hip_Z (deg)")
# axes[2,0].legend()

# # Hip Angle Left    
# # Plot X axis
# axes[0,1].plot(joint_angles["thigh_l"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,1].plot(angles_df["hip_flexion_l"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,1].set_title("Left Hip_X (deg)")
# axes[0,1].legend()
        
# # Plot Y axis
# axes[1,1].plot(joint_angles["thigh_l"][t1:,1], linestyle='-', linewidth=0.7, color="green", label="Captury")
# axes[1,1].plot(angles_df["hip_rotation_l"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[1,1].set_title("Left Hip_Y (deg)")
# axes[1,1].legend()

# # Plot Z axis
# axes[2,1].plot(joint_angles["thigh_l"][t1:,2], linestyle='-', linewidth=0.7, color="blue", label="Captury")
# axes[2,1].plot(angles_df["hip_adduction_l"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[2,1].set_title("Left Hip_Z (deg)")
# axes[2,1].legend()


# # Knee Angle Right    
# # Plot X axis
# axes[0,2].plot(joint_angles["shank_r"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,2].plot(angles_df["knee_angle_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,2].set_title("Right Knee_X (deg)")
# axes[0,2].legend()

# # Knee Angle Left    
# # Plot X axis
# axes[0,3].plot(joint_angles["shank_l"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,3].plot(angles_df["knee_angle_l"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,3].set_title("Left Knee_X (deg)")
# axes[0,3].legend()

# # Ankle Angle Right    
# # Plot X axis
# axes[0,4].plot(joint_angles["foot_r"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,4].plot(angles_df["ankle_angle_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,4].set_title("Right Ankle_X (deg)")
# axes[0,4].legend()

# # Ankle Angle Left    
# # Plot X axis
# axes[0,5].plot(joint_angles["foot_l"][t1:,0], linestyle='-', linewidth=0.7, color="red", label="Captury")
# axes[0,5].plot(angles_df["ankle_angle_l"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[0,5].set_title("Left Ankle_X (deg)")
# axes[0,5].legend()
        
# # Plot Y axis
# # axes[1,4].plot(joint_angles["foot_r"][t1:,1], linestyle='-', linewidth=0.7, color="green", label="Captury")
# axes[1,4].plot(angles_df["subtalar_angle_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[1,4].set_title("Right Ankle_Y (deg)")
# axes[1,4].legend()

# # Plot Z axis
# # axes[2,4].plot(joint_angles["foot_r"][t1:,2], linestyle='-', linewidth=0.7, color="blue", label="Captury")
# axes[2,4].plot(angles_df["mtp_angle_r"], linestyle='-', linewidth=1.2, color="black", label="Vicon")
# axes[2,4].set_title("Right Ankle_Z (deg)")
# axes[2,4].legend()

# fig.suptitle("Joint Angles(Deg)")

# print(angles_df["mtp_angle_r"]) 
# plt.tight_layout()
# plt.show()