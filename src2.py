from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
sigma_v = 0.06     # sd of observation noise
sigma_w = 0.02     # sd of transition noise

f.x = np.array(x_f)
f.F = np.array(phi)
f.H = np.ones((1, 1))
f.P = np.array(sigma_w**2)
f.Q = np.array(sigma_w**2)
f.R = np.array(sigma_v**2)
f.B = np.array(1)


estimations = []
for i in range(1, nSample):
    z = np.array(dt['data'].values[i])
    f.predict(u=c)
    f.update(z)
    estimations.append(f.x)

plt.plot([x[0] for x in estimations])
plt.plot(dt['data'])
plt.axhline(c/(1 - phi), c='r')
plt.show()
