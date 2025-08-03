
import ezc3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# Location of captury csv file
csv_path = "/home/cshah/workspaces/sensorsuit/Captury logs/05_20_2025/sub01_walk_1_2_2.csv"


# If you need to specify an encoding or handle bad lines:
df = pd.read_csv(
    csv_path,
    sep=";",
    header=2,
    encoding="utf-8",       # e.g. latin1, utf-8, etc.
    engine="python",        # sometimes needed for complex separators
)

print(df.shape)

# Hard coded col numbers of information to input
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

# Loop over each joint, pull out its 4 columns, and assign them with nice names
# Rows number from till where to extract data
t1 = 2
t2 = 7500
for joint, (i0,i1,i2,i3) in joint_idxs.items():
    # grab the raw (n_rows × 4) slice
    mat = df.iloc[t1:t2, [i0, i1, i2, i3]].to_numpy()  
    mat = mat.astype(np.float64)
    # pack each row into one object: either a list or small ndarray
    quat_df[joint] = [row.copy() for row in mat]

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


# Convert quaternions to joint angles
# The quaternions reported by captury are already in the locla segment frame
joint_angles = {}
relative_joint_quaternions = {}

for child, parent in joint_heirarchy.items():
    
    relative_joint_quaternions[child] = rotations[child]
  
    joint_angles[child] = relative_joint_quaternions[child].as_euler('xyz', degrees=True)

fig, axes = plt.subplots(3, len(list(joint_angles.keys())), figsize=(20,12), sharex=True)


# plot all the data
for i, joint in enumerate(joint_angles.keys()):
        
    joint_angle = joint_angles[joint]
        
    # Plot X axis
    axes[0,i].plot(joint_angle[:,0], linewidth=1, color="red")
    axes[0,i].set_title(joint + "_X (deg)")
        
    # Plot Y axis
    axes[1,i].plot(joint_angle[:,1], linewidth=1, color="green")
    axes[1,i].set_title(joint + "_Y (deg)")

    # Plot Z axis
    axes[2,i].plot(joint_angle[:,2], linewidth=1, color="blue")
    axes[2,i].set_title(joint + "_Z (deg)")

        
fig.suptitle("Joint Angles(Deg)")
    
plt.tight_layout()
plt.show()