import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver

f = KalmanFilter(dim_x=2, dim_z=1)
f.x = np.array([0., 1.])  # 위치, 속도
f.F = np.array([[1., 1.],
                [0., 1.]])
f.H = np.array([[1., 0.]])
f.R = 5
f.P *= 1000
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

saver = Saver(f)
for z in range(100):
    f.predict()
    f.update([z + np.random.randn() * r_std])
    saver.save()  # save the filter's state

saver.to_array()

print(saver)

plt.figure()
plt.plot(saver.x[:, :])
plt.plot(saver.x_prior[:, :])
plt.plot(saver.mahalanobis)
plt.show()
