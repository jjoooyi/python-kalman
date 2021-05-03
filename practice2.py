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

my_filter = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
numsteps = len(data_dict[1])
f = my_filter
init_state = 47.5
f.x = np.array([[init_state]])
f.F = np.array([[0]])
f.B = np.array([[1]])
f.H = np.array([[1]])

# covariance matrix, P
state_noise = 0.02
f.P = state_noise
# measurement noise, R
measure_noise = 0.8
f.R = np.array([[measure_noise]])
# state uncertainty, Q
f.Q = np.array([[state_noise]])
# control inputs
controls = np.array(data_dict[3])
# get true states
# true_states = np.zeros(numsteps)
# true_states[0] = init_state
# true_states += controls
# # state noise
# true_states += np.random.normal(0, state_noise, numsteps)
# measurements
measurements = [(s + np.random.normal(0, measure_noise)) for s in data_dict[1]]
all_obs = []
estimates = []
num_obs = numsteps
covs = []
for n in range(num_obs):
    my_filter.predict(u=controls[n])
    my_filter.update(measurements[n])
    x = my_filter.x
    res = my_filter.y
    estimates.append(x[0])
    covs.append(my_filter.P[0])

measurements = np.array(measurements)
estimates = np.array(estimates)
# plot results
plt.figure()
plt.plot(range(num_obs), measurements, 'b')
plt.plot(range(num_obs), data_dict[1], 'r')
plt.plot(range(num_obs), estimates, 'g')
plt.legend(('measured', 'true', 'estimates'))
plt.show()
