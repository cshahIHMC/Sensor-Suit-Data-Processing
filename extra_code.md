        
        # raw_quaternions[f"{imu}_ss"] = (t_pose_q_norm[imu].inv() * quat_norm[imu][4422:64412] ).as_quat()
        # raw_quaternions[f"{imu}_ss"] = (quat_norm[imu][4422:64412] ).as_quat()
        
        # # # For the case of the locked pelvis
        # # if "foot" in imu:
        # #     frame_name = "pelvis_2_" + imu
        # #     quat_norm[imu] = quat_norm[imu] 
        # #     # relative_rotations = transforms["Anatomical_2_pelvis"] * t_pose_q_norm["pelvis"].inv() * quat_norm[imu] * transforms["Anatomical_2_pelvis"].inv()
        # #     # relative_rotations = transforms["Anatomical_2_pelvis"] * t_pose_q_norm["pelvis"].inv() * quat_norm[imu] * transforms["Anatomical_2_pelvis"].inv()
        # #     relative_rotations = quat_norm[imu] 
        # # else:
        # #     frame_name = "Anatomical_2_" + imu
        # #     # relative_rotations = transforms[frame_name] * t_pose_q_norm["pelvis"].inv() * quat_norm[imu] * transforms[frame_name].inv()
        # #     # relative_rotations = transforms[frame_name] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name].inv()
            
            
        # #     pelvis_zero = transforms["Anatomical_2_pelvis"] * t_pose_q_norm["pelvis"].inv() * quat_norm["pelvis"] * transforms["Anatomical_2_pelvis"].inv()
        # #     self_zeroing = transforms[frame_name] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name].inv()
            
        # #     # zero the pelvis 
        # #     relative_rotations = pelvis_zero.inv() * self_zeroing
        
        # # frame_name_2 = "Anatomical_2_" + imu
        # # if "foot" in imu:
        # #     # relative_rotations = transforms[frame_name_2] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name_2].inv()
        # #     relative_rotations = transforms[frame_name_2] * t_pose_q_norm[imu].inv() * quat_norm[imu]  * transforms[frame_name_2].inv()
        # #     # relative_rotations = transforms[frame_name] * quat_norm[imu] * t_pose_q_norm[imu].inv()* transforms[frame_name].inv()
        # #     # frame_name = "pelvis_2_" + imu
        # #     # step_1 = t_pose_q_norm["pelvis"] * transforms[frame_name] * quat_norm[imu] * t_pose_q_norm[imu].inv() * transforms[frame_name].inv()
        # #     # relative_rotations = transforms["Anatomical_2_pelvis"] * t_pose_q_norm["pelvis"].inv() * step_1 * transforms["Anatomical_2_pelvis"].inv()
        # # else:
        # #     # relative_rotations = transforms["Anatomical_2_pelvis"] *  quat_norm["pelvis"].inv() * quat_norm[imu]  * transforms["Anatomical_2_pelvis"].inv()
        # #     # relative_rotations = transforms[frame_name_2] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name_2].inv()
        # #     relative_rotations = t_pose_q_norm[imu].inv() * quat_norm[imu]
        # #     # relative_rotations = transforms[frame_name_2] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name_2].inv()
            
        # frame_name = "Anatomical_2_" + imu    
        # relative_rotations = transforms[frame_name] * t_pose_q_norm[imu].inv() * quat_norm[imu] * transforms[frame_name].inv()
