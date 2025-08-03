import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
import Utility

def load_quaternion_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

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

# Transforms to go anatomical frame to sensor frame - all fo these are extrinsic transformations going right to left
def build_transforms():
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

def get_joint_heirarchy():

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
    
    return joint_heirarchy

# this function returns the joint-imu map
def get_joint_imu_map():
    
    joint_imu_map = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu6_quat", 
        "shank_r": "imu4_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu5_quat",
        # "back": "imu3_quat",


        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    
    joint_imu_map_microstrain = {
        "pelvis": "imu2_quat",
        "thigh_r": "imu6_quat", 
        "shank_r": "imu4_quat",
        "thigh_l": "imu1_quat",
        "shank_l": "imu5_quat",
        # "back": "imu3_quat",


    }
    
    joint_imu_map_insole = {
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    return joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole

# Function to extract the imu and insole data and return a list with the entire data and a tpose list
def extract_data(data, joint_imu_map_microstrain, joint_imu_map_insole):
    
    t_pose_q = {limb : np.array(eval(data[f"{imu}"].iloc[0])) for limb, imu in joint_imu_map_microstrain.items()}

    # Extract quat for microstrain
    quat_data = {limb: np.stack(data[f"{imu}"].apply(eval).values) for limb, imu in joint_imu_map_microstrain.items()}
    
    # Extract Insole quaternion data
    for limb, data_key in joint_imu_map_insole.items():
        
        cols = [data_key+"_qw", data_key+"_qx" , data_key+"_qy" , data_key+"_qz" ]
        
        quat_cols = data[cols]
        quat_data[limb] = quat_cols.to_numpy()
        
        t_pose_q[limb] = quat_cols.to_numpy()[0]
  
    return quat_data, t_pose_q
# Apply transformations to the quaternions
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
    
    
    
    # Loop over all IMU's and according to their frames apply the correct transformations
    for imu in data.keys():
        
        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[imu], axis=1, keepdims=True)
        quat_norm[imu] = R.from_quat(data[imu]/quat_magnitude, scalar_first=True)
        
        
        # Step 1: Gravity alignment for T-pose
        t_pose_rot = t_pose_q_norm[imu]
        
        if "foot_r" in imu:
            # 10min in Calibration
            transform_r = [ 0.96805752,  0.24972168, -0.01774981,  0.01163004]
            
            # Raw 
            t_pose_rot =  R.from_quat(transform_r) * t_pose_q_norm[imu]
            
        elif "foot_l" in imu:
            # 10min in Calibration
            transform_l= [ 0.21206732,  0.97710702, -0.01205927, -0.00935059]
            
            # Raw 
            t_pose_rot =  R.from_quat(transform_l) * t_pose_q_norm[imu]
            
            
        
        # Correcting imu wise gravity vectors
        t_pose_z = t_pose_rot.apply([0, 0, 1])
        
        q_gravity_align = R.align_vectors([gravity_vector], [t_pose_z])[0]
        
        # Step 2: Pelvis anatomical frame (after gravity alignment)
        q_anatomical = q_gravity_align * t_pose_rot
        
        # Step 3: Correction quaternion (sensor-to-anatomical)
        q_segment_correction = q_pelvis_anatomical.inv() * q_anatomical
        
        # Corrected pelvis data
        quat_pelvis_aligned_anatomical_data = q_gravity_align_pelvis * quat_norm["pelvis"]
    
       
        
        if "foot_r" in imu:
            # 10min in Calibration
            transform_r = [ 0.96805752,  0.24972168, -0.01774981,  0.01163004]
            
            # Raw 
            # relative_rotations =   R.from_quat(transform_r) * quat_norm[imu] 
            
            # relative_rotations = quat_norm["pelvis"].inv() * relative_rotations
            
            # Zeroing w.r.t itself
            # relative_rotations =  (R.from_quat(transform_r) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_r) * quat_norm[imu] 
            
            # Zeroing w.r.t pelvis
            # relative_rotations =  (t_pose_q_norm["pelvis"].inv() * R.from_quat(transform_r) * t_pose_q_norm[imu]).inv() * quat_norm["pelvis"].inv() * R.from_quat(transform_r) * quat_norm[imu]
            
            # Zeroing w.r.t pelvis and z_axis tilt correction
            A = R.from_quat(transform_r) * quat_norm[imu]   
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_r) * t_pose_q_norm[imu]
            D = t_pose_q_norm["pelvis"].inv() * C
            
            E = D.inv() * B 
                   
            # relative_rotations =  q_gravity_align_pelvis * E
            
            relative_rotations =  E
        elif "foot_l" in imu:

            # 10min in Calibration
            transform_l = [ 0.21206732,  0.97710702, -0.01205927, -0.00935059]
            
            # Raw 
            # relative_rotations = R.from_quat(transform_l) * quat_norm[imu]
            
            # relative_rotations = quat_norm["pelvis"].inv() * relative_rotations
            
            # Zeroing w.r.t itself
            # relative_rotations =      (R.from_quat(transform_l) * t_pose_q_norm[imu]).inv() * R.from_quat(transform_l) * quat_norm[imu]   
            
            # Zeroing w.r.t pelvis
            # relative_rotations =   (t_pose_q_norm["pelvis"].inv() * R.from_quat(transform_l) * t_pose_q_norm[imu]).inv() * quat_norm["pelvis"].inv() * R.from_quat(transform_l) * quat_norm[imu]  
        
            # Zeroing w.r.t pelvis and z_axis tilt correction
            A = R.from_quat(transform_l) * quat_norm[imu]   
            B = quat_norm["pelvis"].inv() * A
            
            C = R.from_quat(transform_l) * t_pose_q_norm[imu]
            D = t_pose_q_norm["pelvis"].inv() * C
            
            E = D.inv() * B
                   
            # relative_rotations = q_gravity_align_pelvis * E
            
            relative_rotations =  E
        # elif "pelvis" in imu:
        #     relative_rotations = t_pose_q_norm["pelvis"].inv() * quat_norm[imu]
        else:
            
            # Raw 
            # relative_rotations = quat_norm[imu] 
            # relative_rotations = quat_norm["pelvis"].inv() * relative_rotations
            
            # Zeroing w.r.t itself
            # relative_rotations = t_pose_q_norm[imu].inv() * quat_norm[imu]
            
            # Zeroing w.r.t pelvis
            # relative_rotations = (t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[imu]).inv() * quat_norm["pelvis"].inv() * quat_norm[imu] 
            
            # Zeroing w.r.t pelvis and z_axis tilt correction
            
            A = quat_norm["pelvis"].inv() * quat_norm[imu]
            B = t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[imu]
            C = B.inv() * A
            # # 
            # # # relative_rotations = q_gravity_align_pelvis * C
            relative_rotations = C

        # relative_rotations = quat_norm[imu]
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
        
        
        # if "pelvis" in child or "thigh" in child:
        #     joint_quaternions[child] = quaternion_data[child]
        # else:
        #     joint_quaternions[child] = quaternion_data[parent].inv() * quaternion_data[child]
            
        joint_angles[child] = joint_quaternions[child].as_euler('xyz', degrees=True)
        
    return joint_angles

def plot_joint_angles(joint_angles, GRF):
    
    fig, axes = plt.subplots(4, len(list(joint_angles.keys())), figsize=(20,12), sharex=True)
    
    print(joint_angles.keys())
    
    for i, joint in enumerate(joint_angles.keys()):
        
        joint_angle = joint_angles[joint]
        
        joint_angle = joint_angles[joint]
        
        # if i == 0:
        #     continue
        
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
        else:
            joint_name = "pelvis"
            
        if i==0:
            alpha = 1
        else:
            alpha = 1
        
        # Plot X axis
        axes[0,i].plot(joint_angle[:,0], linewidth=1, alpha=alpha, color="red")
        # axes[0,i].plot(joint_angle["angle_x"], linewidth=1, color="red")
        axes[0,i].set_title(joint_name + "_X (deg)")
        
        # Plot Y axis
        axes[1,i].plot(joint_angle[:,1], linewidth=1, alpha=alpha, color="green")
        # axes[1,i].plot(joint_angle["angle_y"], linewidth=1, color="green")
        axes[1,i].set_title(joint_name + "_Y (deg)")
    
        # Plot Z axis
        axes[2,i].plot(joint_angle[:,2], linewidth=1, alpha=alpha, color="blue")
        # axes[2,i].plot(joint_angle["angle_z"], linewidth=1, color="blue")
        axes[2,i].set_title(joint_name + "_Z (deg)")
        
        if "_r" in joint:
            axes[3,i].plot(GRF["R_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Right GRF")
        else:
            axes[3,i].plot(GRF["L_insole_force"]/50, linewidth=1, color="black")
            axes[3,i].set_title("Left GRF")
        
        lim=12
        # axes[0,i].set_ylim(-lim,lim)
        # axes[1,i].set_ylim(-lim,lim)
        # axes[2,i].set_ylim(-lim,lim)
        # axes[3,i].set_ylim(-lim,lim)
        

        
    fig.suptitle("Joint Angles(Deg)")
    
    plt.tight_layout()
    plt.show()
    
    
def main():
    
    csv_path = "/home/cshah/workspaces/sensorsuit/SensorSuit-logs/07_09_2025_Nicole/07_09_2025_GA_Tech_sensor_Config/07_09_SQ_10_01.csv"
    
    # Load the data to a csv
    data = load_quaternion_data(csv_path=csv_path)
    
    force = ["L_insole_force", "R_insole_force"]
    insole_force_data = data[force]
    
    # Build the Transforms from one frame to the other
    body_transforms = build_transforms()
    joint_heirarchy = get_joint_heirarchy()
    joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole = get_joint_imu_map()
    
    quaternion_data, t_pose_quat = extract_data(data=data, 
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
    