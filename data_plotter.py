import os
import pandas as pd
import re
import Utility 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# This file is responsible for plotting the data from the IMU
# The data is collected in the form of a csv file


# read in CSV file from specified path
while (True):
 
    # file_path for walking data
    file_path = "/home/cshah/workspaces/sensorsuit/logs/05_08_2025/05_08_2025_start_0_walk_test.csv"

    if os.path.isfile(file_path):

        break
    else:
        print('File does not exist...')
        break


data_df = pd.read_csv(file_path)


# determine the number of unique IMUs there are
imu_identifiers = set()
for col in data_df.columns:
    match = re.match(r'(imu\d+)', col) # match 'imu followed by digits
    if match:
        imu_identifiers.add(match.group(1))


# count number of unique IMUs
num_imus = len(imu_identifiers)

# correct time columns of IMUs
for i in range(1, num_imus + 1):

    # check if the first value is zero. If so then subtract from the second value of timestamp
    # if data_df[f'imu{i}_timestamp'].iloc[0] <= 0.001:
    #     data_df[f'imu{i}_time'] = data_df[f'imu{i}_timestamp'] - data_df[f'imu{i}_timestamp'].iloc[1]
    #     data_df[f'imu{i}_time'].iloc[0] = 0
    # else:
    #     data_df[f'imu{i}_time'] = data_df[f'imu{i}_timestamp'] - data_df[f'imu{i}_timestamp'].iloc[0]
    #     data_df[f'imu{i}_time'].iloc[0] = 0

    data_df[f'imu{i}_time'] = data_df[f'imu{i}_timestamp'] - data_df[f'imu{i}_timestamp'].iloc[1]
    data_df[f'imu{i}_time'].iloc[0] = 0

sensor_suit_imu_data = {
     "Back Pack IMU": "imu3",
     "Pelvis": "imu2",
     "Right Thigh": "imu5",
     "Left Thigh": "imu1",
     "Right Shank": "imu6",
     "Left Shank": "imu4",
     "Right Foot": "R_insole",
     "Left Foot": "L_insole"
}


sensor_suit_GRF_data = {
     "R_GRF": "R_insole",
     "L_GRF": "L_insole"
}

# for i in range(len(data_df["Device Time"])):
#     device = data_df["Device Time"][i]
#     imu1 = data_df["imu1_time"][i]
#     imu1_timestamp = data_df["imu1_timestamp"][i]


#     print(f"Device time {device}, imu 1 {imu1}, imu 1 timestamp {imu1_timestamp}")

#     if i==3:
#         break

for col in data_df.columns:
    print(col)

# print(data_df.columns)
# Utility.plot_imu_data(data_df, sensor_suit_imu_data)

# Device time analysis

plt.figure(figsize=(6,5))

plt.plot(data_df['Device Time'], data_df["imu1_time"], label="Device Time vs imu1")
plt.plot(data_df['Device Time'], data_df["imu2_time"], label="Device Time vs imu2")
plt.plot(data_df['Device Time'], data_df["imu3_time"], label="Device Time vs imu3")
plt.plot(data_df['Device Time'], data_df["imu4_time"], label="Device Time vs imu4")
plt.plot(data_df['Device Time'], data_df["imu5_time"], label="Device Time vs imu5")
plt.plot(data_df['Device Time'], data_df["imu6_time"], label="Device Time vs imu6")

plt.title("Device time vs IMU time")
plt.xlabel(" IMU time stamp zeroed to first timestep (s)")
plt.ylabel("Device Time (s)")
plt.legend()

plt.figure(figsize=(6,5))

diff_deviceTime_imu1 = data_df['Device Time'] - data_df["imu1_time"]
diff_deviceTime_imu2 = data_df['Device Time'] - data_df["imu2_time"]
diff_deviceTime_imu3 = data_df['Device Time'] - data_df["imu3_time"]
diff_deviceTime_imu4 = data_df['Device Time'] - data_df["imu4_time"]
diff_deviceTime_imu5 = data_df['Device Time'] - data_df["imu5_time"]
diff_deviceTime_imu6 = data_df['Device Time'] - data_df["imu6_time"]


plt.plot(diff_deviceTime_imu1, label="Device Time - imu1")
plt.plot(diff_deviceTime_imu2, label="Device Time - imu2")
plt.plot(diff_deviceTime_imu3, label="Device Time - imu3")
plt.plot(diff_deviceTime_imu4, label="Device Time - imu4")
plt.plot(diff_deviceTime_imu5, label="Device Time - imu5")
plt.plot(diff_deviceTime_imu6, label="Device Time - imu6")

plt.legend()

plt.title("Device time - imu time")




dt_deviceTime = np.diff(data_df['Device Time'])
dt_imu1Time = np.diff(data_df["imu1_time"])
dt_imu2Time = np.diff(data_df["imu2_time"])
dt_imu3Time = np.diff(data_df["imu3_time"])
dt_imu4Time = np.diff(data_df["imu4_time"])
dt_imu5Time = np.diff(data_df["imu5_time"])
dt_imu6Time = np.diff(data_df["imu6_time"])


plt.figure(figsize=(6,5))

# plt.boxplot([dt_deviceTime, dt_imu1Time, dt_imu2Time, dt_imu3Time, dt_imu4Time, dt_imu5Time, dt_imu6Time], positions=[1,2,3,4,5,6, 7] ,vert=True, patch_artist=True, 
#             boxprops=dict(facecolor='lightblue', color='black'),
#             whiskerprops=dict(color='blue', linewidth=2),
#             capprops=dict(color='black', linewidth=2),
#             flierprops=dict(marker='o', markerfacecolor='red', markersize=1, linestyle='none'))

plt.plot( dt_deviceTime, label="Device Time - imu1")
plt.plot( dt_imu1Time, label="Device Time - imu2")
plt.plot( dt_imu2Time, label="Device Time - imu3")
plt.plot( dt_imu3Time, label="Device Time - imu4")
plt.plot( dt_imu4Time, label="Device Time - imu5")
plt.plot( dt_imu5Time, label="Device Time - imu6")
plt.plot( dt_imu6Time, label="Device Time - imu6")

# plt.xlim(0,8)

Utility.calculateStats(dt_deviceTime, "Device Time")
Utility.calculateStats(dt_imu1Time, "imu 1")
Utility.calculateStats(dt_imu2Time, "imu 2")
Utility.calculateStats(dt_imu3Time, "imu 3")
Utility.calculateStats(dt_imu4Time, "imu 4")
Utility.calculateStats(dt_imu5Time, "imu 5")
Utility.calculateStats(dt_imu6Time, "imu 6")

plt.show()








