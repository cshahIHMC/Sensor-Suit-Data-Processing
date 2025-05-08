import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys


# Load quaternion data from CSV
def load_quaternion_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Convert quaternion to rotation matrix
def quaternion_to_matrix(q):
    return R.from_quat(q).as_matrix()

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
    
    transforms["Animation_2_shank_l"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(np.pi/2) ).as_quat()
    transforms["Animation_2_shank_r"] = R.from_matrix(get_R_y(-np.pi/2) @ get_R_x(-np.pi/2) ).as_quat()
    
    ## This s the foot frame when the IMU is on the tongue
    transforms["Animation_2_foot_l"] = R.from_matrix(get_R_z(np.pi)).as_quat()
    transforms["Animation_2_foot_r"] = R.from_matrix(get_R_z(np.pi)).as_quat()
    
    ## Pelvis 2 foot if foot is on the Toe
    # transforms["pelvis_2_foot"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    
    ## Pelvis 2 foot if foot is on the Heel
    # transforms["pelvis_2_foot_r"] = R.from_matrix(get_R_x(-np.pi/2)).as_quat()
    # transforms["pelvis_2_foot_l"] = R.from_matrix(get_R_x(np.pi/2)).as_quat()
    
    # transforms["pelvis_2_foot_r"] = R.from_matrix(get_R_x(np.pi) @ get_R_y(np.pi/2)).as_quat()
    # transforms["pelvis_2_foot_l"] = R.from_matrix(get_R_x(np.pi) @ get_R_y(np.pi/2)).as_quat()
    
    
    transforms["pelvis_2_foot_r"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    transforms["pelvis_2_foot_l"] = R.from_matrix(get_R_y(np.pi/2)).as_quat()
    
    return transforms


# Apply transformations - key to unlocked pelvis
def transform_quaternions_unlocked_pelvis(data, t_pose_q, transforms):
    
    # Data to return
    transformed_data = {}
    
    A = R.from_quat(transforms["Animation_2_pelvis"])
    
    # Normalize t-pose
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}
    # Compute tpose relative to pelvis frame
    t_pose_q_norm_rel_pelvis = {imu: t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[imu] for imu in t_pose_q_norm.keys()}
    

    
    for imu, raw in data.items():
        
        # Normalize the raw data
        q_raw = R.from_quat(raw / np.linalg.norm(raw, axis=1, keepdims=True), scalar_first=True)
        
        if "foot" in imu:
            
            frame_name = "pelvis_2_" + imu            
            q_raw =  t_pose_q_norm["pelvis"] * R.from_quat(transforms[frame_name]) * q_raw * t_pose_q_norm[imu].inv() * R.from_quat(transforms[frame_name]).inv()
       
        # pelvis-local
        q_local = t_pose_q_norm["pelvis"].inv() * q_raw

        # animation-local
        q_anim = A.inv() * q_local * A
        # delta from T-pose
        # q_rel  = t_pose_q_norm_rel_pelvis[imu].inv() * q_anim
            
        
        transformed_data[imu] = q_anim.as_quat()
        
    return transformed_data
        
    
# Apply transformations
def transform_quaternions_locked_pelvis(data, t_pose_q, transforms):
    transformed_data = {}

    # Get the the initial quaternions from the t pose and normalize them
    t_pose_q_norm = {imu: R.from_quat(q/np.linalg.norm(q), scalar_first=True) for imu, q in t_pose_q.items()}
    t_pose_q_norm_rel_pelvis = {imu: t_pose_q_norm["pelvis"].inv() * t_pose_q_norm[imu] for imu in t_pose_q_norm.keys()}
    
    measured_transform = {}
    ### THis gives you the exact transform to go from animation to respective imu frame
    for imu in t_pose_q_norm_rel_pelvis.keys():
        
        print(imu)
        q = R.from_quat(transforms["Animation_2_pelvis"]).inv() * t_pose_q_norm_rel_pelvis[imu] * R.from_quat(transforms["Animation_2_pelvis"])
        deg = q.as_euler("xyz", degrees=True)
        measured_transform[imu] = q.as_quat()
        print(deg)

    quaternions = {}
    

    # Loop over all IMU's and according to their frames apply the correct transformations
    for imu in data.keys():

        # Normalize all imu's
        quat_magnitude = np.linalg.norm(data[imu], axis=1, keepdims=True)
        quaternions[imu] = R.from_quat(data[imu]/quat_magnitude, scalar_first=True)
        

        
        # if "foot" not in imu and "pelvis" not in imu:
        #     name = "Animation_2_" + imu
        #     relative_rotations =  R.from_quat(measured_transform[imu]).inv() * t_pose_q_norm[imu].inv() * quaternions[imu] 
        #     # relative_rotations = pelvis_zeroing.inv() * relative_rotations
        #     # relative_rotations = R.from_quat(transforms["Animation_2_pelvis"]).inv() * quaternions["pelvis"].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])
        # elif "pelvis" in imu:
        #     relative_rotations = R.from_quat(transforms["Animation_2_pelvis"]).inv() * t_pose_q_norm[imu].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])
        #     pelvis_zeroing = relative_rotations
        # else:
        #     relative_rotations =  pelvis_zeroing.inv() * t_pose_q_norm[imu].inv() * quaternions[imu]
            
            
        ############################# section for locked pelvis #################################

        # Move the x sensor imu's from their own world reference frame to the NED world frame consistent with the microstrain
        if "foot" in imu:
            
            print(imu)
            print("Did pre processing on this data")


            # q align will take the foot imu readings move it to the body frame, then pelvis frame and then to the world (NED frame)
            # q align = q_pelvis_t_pose * q_pelvis_2_foot_transform.inv() * q_foot_t_pose_inv
            
            # Stuff that worked 
            
            # Taking it to the NED frame
            # Feet IMU at the Toe Position
            frame_name = "pelvis_2_" + imu
            quaternions[imu] = t_pose_q_norm["pelvis"] * R.from_quat(transforms[frame_name]) * quaternions[imu] * t_pose_q_norm[imu].inv() * R.from_quat(transforms[frame_name]).inv()
            # quaternions[imu] =  t_pose_q_norm["pelvis"] * R.from_quat(transforms["pelvis_2_foot"]) * t_pose_q_norm[imu].inv() * quaternions[imu] * R.from_quat(transforms["pelvis_2_foot"]).inv() 
            
            
            # Feet IMU at the Heel Position
            # frame_name = "pelvis_2_" + imu
            # print(frame_name)
            # # quaternions[imu] = t_pose_q_norm["pelvis"] * R.from_quat(transforms[frame_name]) * quaternions[imu] * t_pose_q_norm[imu].inv() * R.from_quat(transforms[frame_name]).inv()
            
            # quaternions[imu] = quaternions[imu] * t_pose_q_norm[imu]
                
            

        # Move all IMU's from the world to the pelvis frame and change the reference frame to the pelvis frame and rotate everything to the animation frame
        if imu != "pelvis" and "foot" not in imu:
            # Below transformations put all of the imu's in the pelvis frame and then in the animation frame
            # R_anim_NED = t_pose_q_norm["pelvis"] * R.from_quat(transforms["Animation_2_pelvis"])
            relative_rotations = R.from_quat(transforms["Animation_2_pelvis"]).inv() * quaternions["pelvis"].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])
            # relative_rotations = R.from_quat(transforms["Animation_2_" + imu]).inv() * t_pose_q_norm[imu].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_" + imu])
        elif "foot" in imu:
            # relative_rotations =  R.from_quat(transforms["Animation_2_pelvis"]).inv() * quaternions["pelvis"].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])

            relative_rotations = R.from_quat(transforms["Animation_2_pelvis"]).inv() * quaternions["pelvis"].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])
            print(imu)
            print("Final Processing")
            
            # relative_rotations = quaternions[imu]
            
            # relative_rotations = quaternions[imu]
        else :
            relative_rotations = R.from_quat(transforms["Animation_2_pelvis"]).inv() * quaternions["pelvis"].inv() * quaternions[imu] * R.from_quat(transforms["Animation_2_pelvis"])
        
        #################################################################################################################
        
        ########################################### Section for unlocked pelvis ##############################################
        
        # print(imu)
        # # Zero all the imu's w.r.t itself and transfer to the pelvis frame and then to animation frame
        # if "foot" not in imu and "pelvis" not in imu:
        #     frame_name = "Animation_2_" + imu
        #     relative_rotations =   R.from_quat(transforms[frame_name]).inv() * t_pose_q_norm[imu].inv() * quaternions[imu] * R.from_quat(transforms[frame_name])
        # elif "pelvis" in imu:
        #     frame_name = "Animation_2_" + imu
        #     relative_rotations =   R.from_quat(transforms[frame_name]).inv() * t_pose_q_norm[imu].inv() * quaternions[imu] * R.from_quat(transforms[frame_name])
    
        # else:
        #     frame_name = "Animation_2_" + imu
        #     relative_rotations =   R.from_quat(transforms[frame_name]) * quaternions[imu] * t_pose_q_norm[imu].inv() * R.from_quat(transforms[frame_name]).inv()
            

        ############################################################################################################################
        # if imu == "foot_l" :


        #     print("Quaternion Data")
            
        #     tpose_angles = t_pose_q_norm[imu].as_euler('xyz', degrees=True)
        #     print(tpose_angles)

        #     X, Y, Z = [], [], []
        #     X2, Y2, Z2 = [], [], []
        #     for i in range(len(quaternions[imu])):
        #         x, y, z = (quaternions["pelvis"][i]).as_euler('xyz', degrees=True)
        #         euler_angles = relative_rotations[i].as_euler('xyz', degrees=True)

        #         # Unwrap the angles to remove discontinuities
        #         # euler_angles = np.unwrap(euler_angles, axis=0)

        #         X.append(x)
        #         Y.append(y)
        #         Z.append(z)

        #         X2.append(euler_angles[0])
        #         Y2.append(euler_angles[1])
        #         Z2.append(euler_angles[2])

        #     # Create subplots
        #     fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        #     # ax[0].plot(X, color='r')
        #     ax[0].plot(X2, color='r', alpha=0.5)
        #     ax[0].set_ylabel("X (degrees)")
        #     ax[0].set_title("Euler Angle Transformations")

        #     # ax[1].plot(Y, color='g')
        #     ax[1].plot(Y2, color='g', alpha=0.5)
        #     ax[1].set_ylabel("Y (degrees)")

        #     # ax[2].plot(Z, color='b')
        #     ax[2].plot(Z2, color='b', alpha=0.5)
        #     ax[2].set_ylabel("Z (degrees)")
        #     ax[2].set_xlabel("Time Step")


        #     plt.tight_layout()
        #     plt.show()







        transformed_data[imu] = relative_rotations.as_quat()


    return transformed_data


# Extract joint positions using rotation matrices
def get_joint_positions(transformed_data, limb_structure, segment_lengths):
# def get_joint_positions(transformed_data, limb_structure, segment_lengths, t1, t2):
    positions = {imu: [] for imu in limb_structure.keys()}
    positions["pelvis"] = []
    for i in range(len(transformed_data["pelvis"])):
    # for i in range(t1*150, t2*150):
        pos = {"pelvis": np.array([0, 0, 0])}  # Root at the origin
        for segment, (start, end) in limb_structure.items():
            if "pelvis" in segment:
                ## If pelvis assuming it to be fixed
                # rot = R.from_matrix(np.eye(3))
                
                ## If pelvis is not fixed
                rot = R.from_quat(transformed_data[start][i])
            else:
                # rot_matrix = quaternion_to_matrix(transformed_data[end][i])
                rot = R.from_quat(transformed_data[end][i])
                
            # rot_matrix = quaternion_to_matrix(transformed_data[end][i])

            
            if "_r" in segment:
                if "pelvis" in segment:
                    # direction = np.dot(rot_matrix, np.array([0, -1 , 0])) * segment_lengths[segment]
                    
                    direction = rot.apply(np.array([0, -1 , 0])) * segment_lengths[segment]
                elif "thigh" in segment or "shank" in segment:
                    # direction = np.dot(rot_matrix, np.array([0, 0, -1])) * segment_lengths[segment]
                    
                    direction = rot.apply(np.array([0, 0, -1])) * segment_lengths[segment]
                    
                elif "foot" in segment:
                    direction = rot.apply(np.array([1, 0, 0])) * segment_lengths[segment]
                    
            elif "_l" in segment:
                if "pelvis" in segment:
                    # direction = np.dot(rot_matrix, np.array([0, 1, 0])) * segment_lengths[segment]
                    
                    direction = rot.apply(np.array([0, 1, 0])) * segment_lengths[segment]
                elif "thigh" in segment or "shank" in segment:
                    # direction = np.dot(rot_matrix, np.array([0, 0, -1])) * segment_lengths[segment]
                    
                    direction = rot.apply(np.array([0, 0, -1])) * segment_lengths[segment]
                    # p = np.array([0, 0, -1]) * segment_lengths[segment]
                    # direction = rot_matrix.apply(p)
                elif "foot" in segment:
                    direction = rot.apply(np.array([1, 0, 0])) * segment_lengths[segment]
            else:
                # This one refers to the back
                direction = rot.apply(np.array([0, 0, 1])) * segment_lengths[segment]

            pos[end] = pos[start] + direction

        for imu in positions:
            positions[imu].append(pos[imu])
    return positions


# Extracting the Insole data
def extract_insole_imu_data(quaternion_data, all_data, tpose_data, t_pose_q):
    
    
    insoles_keys = {
        "foot_r": "R_insole",
        "foot_l": "L_insole"
    }
    
    for limb, data_key in insoles_keys.items():
        
        cols = [data_key+"_qw", data_key+"_qx" , data_key+"_qy" , data_key+"_qz" ]
        
        quat_cols = all_data[cols]
        quaternion_data[limb] = quat_cols.to_numpy()
        
        tpose_quat_cols = tpose_data[cols]
        
        t_pose_q[limb] = tpose_quat_cols.to_numpy()[0]
        
        
    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres...'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = sum(x_limits) / 2
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = sum(y_limits) / 2
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = sum(z_limits) / 2

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

     
# Function to just display the calculate limb positions 
def animate_motion_3d(positions, limb_structure, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    set_axes_equal(ax)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    ax.view_init(elev=20, azim=0)
    
    # Create a dictionary to store line objects for each limb segment with colors
    lines = {}
    for segment in limb_structure:
        color = 'red' if '_l' in segment else 'blue' if '_r' in segment else 'black'
        lines[segment] = ax.plot([], [], [], 'o-', lw=2, color=color)[0]

    def update(frame):
        for segment, (start, end) in limb_structure.items():
            x_data = [positions[start][frame][0], positions[end][frame][0]]
            y_data = [positions[start][frame][1], positions[end][frame][1]]
            z_data = [positions[start][frame][2], positions[end][frame][2]]
            lines[segment].set_data(x_data, y_data)
            lines[segment].set_3d_properties(z_data)
        return lines.values()

    ani = animation.FuncAnimation(fig, update, frames=len(positions["pelvis"]), interval=1, blit=True)
    
    # ani.save(file_name, writer='ffmpeg', fps=150, dpi=500, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    # print(f"Saved animation to {file_name}")
    
    ani.save("walking_animation_2.gif", writer='pillow', fps=150)
    
    plt.show()

def animate_motion_3d_pyqtgraph(positions, limb_structure):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = gl.GLViewWidget()
    window.setWindowTitle('3D Limb Motion')
    window.setGeometry(0, 0, 800, 600)
    window.setBackgroundColor('w') 
    window.opts['distance'] = 2  # Camera distance
    window.setCameraPosition(elevation=20, azimuth=-90)
    window.show()
    
    

    # Create a dict of line segments
    lines = {}
    spheres = {}
    for segment in limb_structure:
        color = (1, 0, 0, 1) if '_l' in segment else (0, 0, 1, 1) if '_r' in segment else (0.5, 0.5, 0.5, 1)
        plt = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=5, antialias=True)
        plt.setGLOptions("translucent")
        lines[segment] = plt
        window.addItem(plt)
        
        # Create circles (spheres) at the joint positions
        sphere_color = (1, 0, 0, 1) if '_l' in segment else (0, 0, 1, 1) if '_r' in segment else (0.5, 0.5, 0.5, 1)
        sphere = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=sphere_color, size=10)  # Adjust size as needed
        sphere.setGLOptions("translucent")
        spheres[segment] = sphere
        window.addItem(sphere)
        
    
    

    frame_idx = 0
    num_frames = len(positions["pelvis"])

    def update():
        nonlocal frame_idx
        
        if frame_idx >= num_frames:
            timer.stop()
            print("Done")
            app.quit()   
            return
        
        for segment, (start, end) in limb_structure.items():
            start_pos = positions[start][frame_idx]
            end_pos = positions[end][frame_idx]
            lines[segment].setData(pos=np.array([start_pos, end_pos]))
            
            # Set the sphere (circle) at the end joints
            spheres[segment].setData(pos=np.array([start_pos, end_pos]))  # Update both joints (start and end)
        
        
        frame_idx = (frame_idx + 1)  # Loop animation

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 // 200)  # ~150 FPS

    sys.exit(app.exec_())
    
# Define limb structure and segment lengths
limb_keys = {
    
    # "pelvis": "imu2_quat",
    # "back":  "imu3_quat",
    # "thigh_l": "imu1_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu4_quat",
    # "shank_r": "imu5_quat"
    
    # "foot_l": 5,
    # "foot_r": 6,
    
    # "pelvis": "imu2_quat",
    # "back": "imu3_quat",
    # "thigh_l": "imu1_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu5_quat",
    # "shank_r": "imu4_quat",
    
    
    # #04_04_2025 Collection
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu1_quat",
    # "back": "imu3_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu4_quat",
    # "shank_r": "imu5_quat",
    
    
        
    #Foot Zeroing Collection
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu4_quat",
    # "back": "imu1_quat",
    # "thigh_r": "imu3_quat", 
    # "shank_l": "imu6_quat",
    # "shank_r": "imu5_quat",
    
    # 04_09_2025_Collection - Imu position on toe
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu1_quat",
    # "back": "imu3_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu4_quat",
    # "shank_r": "imu5_quat",
    
    # 04_09_2025_Collection - Imu position on Heel
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu1_quat",
    # "back": "imu4_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu3_quat",
    # "shank_r": "imu5_quat",
    
    
        
    # 123 - Ankle Validation
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu5_quat",
    # "back": "imu6_quat",
    # "thigh_r": "imu1_quat", 
    # "shank_l": "imu3_quat",
    # "shank_r": "imu4_quat",
    
    # 04_21_2025_ Collection - Imu position on heel - Just walking
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu1_quat",
    # "back": "imu3_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu4_quat",
    # "shank_r": "imu5_quat",
    
    # 04_28_2025_ Collection - Imu position on heel - 2 walking set collection
    
    # "pelvis": "imu2_quat",
    # "thigh_l": "imu1_quat",
    # "back": "imu3_quat",
    # "thigh_r": "imu6_quat", 
    # "shank_l": "imu5_quat",
    # "shank_r": "imu4_quat",
    
    # Single joint tests
        # "pelvis": "imu2_quat",
        # "thigh_l": "imu1_quat",
        # "back": "imu3_quat",
        # "thigh_r": "imu6_quat", 
        # "shank_l": "imu5_quat",
        # "shank_r": "imu4_quat",
        
    # 05_08_2025 - new thigh imu location + joint level collection + walk
    
    "pelvis": "imu2_quat",
    "thigh_r": "imu5_quat", 
    "shank_r": "imu6_quat",
    "thigh_l": "imu1_quat",
    "shank_l": "imu4_quat",
    "back": "imu3_quat",
    
    
    
    



}

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
segment_lengths = {"back": 0.3, "pelvis_l": 0.2, "pelvis_r":0.2, "thigh_l":0.4, "thigh_r": 0.4, "shank_l" : 0.4, "shank_r":0.4, "foot_l": 0.075, "foot_r":0.075} ## "pelvis_r": 0.2, "thigh_l": 0.4, "thigh_r": 0.4} ## "shank_l": 0.4, "shank_r": 0.4} ## "foot_l": 0.2, "foot_r": 0.2}


# Main execution
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/06_03_2025/06_03_2025_10min_walk_stop_test.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/insole_moved_2.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_04_2025/04_04_2025_10_min_trial_1.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_04_2025/04_04_2025_tpose.csv"



# t_pose_csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_04_2025/04_04_2025_tpose.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_04_2025/04_04_2025_10_min_trial_2.csv"

# t_pose_csv_path = "/home/cshah/workspaces/sensorsuit/logs/12_tpose.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/05_01_2025/05_01_2025_feet_on_toes_2.csv"

# # Full data collection
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_09_2025/04_09_2025_trial_1.csv"

# Feet Validation Data Collection
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/123_right_foot_inside_out.csv"

# 04_21_2025 -Foot and just walking collection
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/04_28_2025/04_28_2025_walk_test_1.csv"
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/05_01_2025/05_01_2025_sj_thigh_45_back.csv"


# 05_08_2025 - SJ  + Test cases + Walk_test - New thigh imu location - IMP DATAset
# csv_path = "/home/cshah/workspaces/sensorsuit/logs/05_08_2025/05_08_2025_6_feet_on_toes_start_right_turn_l.csv"
csv_path = "/home/cshah/workspaces/sensorsuit/logs/05_08_2025/05_08_2025_6_feet_on_toes_turn_r_turn_l.csv"





# Extracting the data from csv
# tpose_data = load_quaternion_data(t_pose_csv_path)
data = load_quaternion_data(csv_path)


# Building the transforms
body_transforms = build_transforms()


# Populaitng the microstrain imu data
# t_pose_q = {limb : np.array(eval(tpose_data[f"{imu}"].iloc[50])) for limb, imu in limb_keys.items()}
t_pose_q = {limb : np.array(eval(data[f"{imu}"].iloc[0])) for limb, imu in limb_keys.items()}

quaternion_data = {limb: np.stack(data[f"{imu}"].apply(eval).values) for limb, imu in limb_keys.items()}

# Populating the xsensor IMU data
# extract_insole_imu_data(quaternion_data, data, tpose_data, t_pose_q)
extract_insole_imu_data(quaternion_data, data, data, t_pose_q)



# Performing all the appropriate transformations to bring everything to the appropriate frames
transformed_data = transform_quaternions_locked_pelvis(quaternion_data, t_pose_q, body_transforms)
# transformed_data = transform_quaternions_unlocked_pelvis(quaternion_data, t_pose_q, body_transforms)


# Getting all the joint positions from the transformed data
positions = get_joint_positions(transformed_data, limb_structure, segment_lengths)
# positions = get_joint_positions(transformed_data, limb_structure, segment_lengths, 612, 618)

# # Animate the calculated positions

file_name = "walking_animation.mp4"
# animate_motion_3d(positions, limb_structure, file_name)
animate_motion_3d_pyqtgraph(positions, limb_structure)