from matplotlib import pyplot as plt
import numpy as np

data = []
with open('sensor_log01.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            data.append(float(line.split("|")[3]))


def do_averaging_7(data):
    return [(sum(y) + sum(y[1:-1]) + sum(y[2:-2]) + 7) // 15
            for y in zip(data[:-6], data[1:-5], data[2:-4], data[3:-3], data[4:-2], data[5:-1], data[6:])]


def moving_average1(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_average2(a, n=3):  # "n=3" indicates the default value
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret /= n
    # Masking initial a few values to prevent weird values
    ret[:n-1] = ret[n-1]
    return ret


graph = data

# avg = do_averaging_7(data)
# avg = moving_average1(data, n=7)
avg = moving_average2(data[6:], n=7)

fig, ax = plt.subplots()
ax.plot(range(len(data)), data, "blue")
ax.plot(range(3, 3+len(avg)), avg, color="red")
ax.set_ylabel('Raw')
plt.tight_layout()
plt.show()
