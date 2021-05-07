import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt


data_dict = {}
for i in range(1, 6):
    with open("sensor_log0{}.txt".format(i), "r") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line:
                data.append(float(line.split("|")[3]))
        data_dict[i] = data


f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
numsteps = len(data_dict[1])
init_state = 47.5

# x - 상태 측정 벡터
f.x = np.array([[init_state]])
# 상태 전이 행렬 F
f.F = np.array([[0]])
f.B = np.array([[1]])
# 측정 기능 H
f.H = np.array([[1]])
# 공분산 행렬 P
state_noise = 0.02
f.P = state_noise
# 측정 노이즈 행렬 R
measure_noise = 0.8
f.R = np.array([[measure_noise]])
# 프로세스 노이즈 행렬 Q
f.Q = np.array([[state_noise]])

# control inputs
controls = np.array(data_dict[3][9:])
# 측정값
measurements = [(s + np.random.normal(0, measure_noise)) for s in data_dict[1]]

all_obs = []
estimates = []
covs = []
for n in range(numsteps):
    f.predict(u=controls[n])
    f.update(measurements[n])
    x = f.x
    res = f.y
    estimates.append(x[0])
    covs.append(f.P[0])

measurements = np.array(measurements)
estimates = np.array(estimates)

# plot results
plt.figure()
plt.plot(range(numsteps), measurements, 'b')
plt.plot(range(numsteps), data_dict[1], 'r')
plt.plot(range(numsteps), estimates, 'g')
plt.legend(('measured', 'true', 'estimates'))
plt.show()
