# https://dsp.stackexchange.com/questions/48911/how-to-use-kalman-filter-for-altitude-prediction-based-on-barometer-data

import numpy as np
import matplotlib.pyplot as plt
import random

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

random.seed(65537)

# Standard deviation of simulated sensor data
std_a = 0.075
std_h = 0.42

# Earth gravity
g_n = 9.80665

# Least Squares Linear Regression


class LinearRegression:
    def __init__(self, N):
        self.N = N
        self.c = 0
        self.x = np.zeros(N)
        self.y = np.zeros(N)

    def update(self, x, y):
        self.x[1:] = self.x[:-1]
        self.y[1:] = self.y[:-1]

        self.x[0] = x
        self.y[0] = y
        if (self.c < self.N):
            self.c += 1

    def slope(self):

        if (self.c < self.N):
            return 0

        sum_x = np.sum(self.x)
        sum_y = np.sum(self.y)
        sum_xx = np.sum(self.x * self.x)
        sum_xy = np.sum(self.x * self.y)
        sum_yy = np.sum(self.y * self.y)

        a = (self.N * sum_xy - sum_x * sum_y) / \
            (self.N * sum_xx - sum_x * sum_x)

        return a


# Sensor sample period, s
dt = 0.05

# Linear regression size
M = 96
lr = LinearRegression(M)

kf = KalmanFilter(dim_x=3, dim_z=2)
kf.H = np.array([[1, 0, 0], [0, 0, 1]])
kf.F = np.array([[1, dt, dt * dt * 0.5], [0, 1, dt], [0, 0, 1]])

# initial process covariance
kf.P = np.array([[std_h * std_h, 0, 0], [0, 1, 0], [0, 0, std_a * std_a]])

# Process noise matrix
std = 0.004
var = std * std
kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=var)

# Measurement covariance
kf.R *= np.array([[std_h * std_h, 0], [0, std_a * std_a]])

n = 300
r_n = 1.0 / n

t = np.zeros(n)
h_sim = np.zeros(n)
v_sim = np.zeros(n)
a_sim = np.zeros(n)
measured_h = np.zeros(n)
measured_a = np.zeros(n)
v_est_lr = np.zeros(n)
v_est_kf = np.zeros(n)

v0 = 0
v = v0
h = 0

for i in range(n):
    t[i] = (i * dt)
    a = 1.0 / 32 * g_n * np.sin(4 * np.pi * i * r_n)
    a_sim[i] = (a)
    v += a * dt
    v_sim[i] = v
    h += v * dt + (a * dt * dt) / 2
    h_sim[i] = h
    measured_a[i] = a + random.gauss(0, std_a)
    measured_h[i] = h + random.gauss(0, std_h)

# Compute the speed estimations

for i in range(n):
    v_est_kf[i] = kf.x[1]
    kf.predict()
    kf.update(np.array([[measured_h[i]], [measured_a[i]]]))

    lr.update(i * dt, measured_h[i])
    v_est_lr[i] = lr.slope()

# Plot the results

plt.figure(1, figsize=(8, 12), dpi=80)

plt.subplot(311)
plt.axis([0, n * dt, -0.75, 0.75])
plt.plot(t, measured_a, 'y+')
plt.plot(t, a_sim, 'r')
plt.title('Vertical acceleration - gravity, m/s^2')
plt.legend(('Measured', 'True (simulation)'), loc='best')

plt.subplot(312)
plt.axis([0, n * dt, -1, 7])
plt.plot(t, measured_h, 'c+')
plt.plot(t, h_sim, 'b')
plt.title('Altitude, m')
plt.legend(('Measured', 'True (simulation)'), loc='best')

plt.subplot(313)
plt.axis([0, n * dt, -0.5, 1.0])
plt.plot(t, v_sim, 'b')
plt.plot(t, v_est_kf, 'g')
plt.plot(t[M:], v_est_lr[M:], 'r')
plt.title('Vertical speed, m/s')
plt.legend(('True (simulation)', 'Kalman filter',
           'Linear Regression'), loc='best')
plt.show()
