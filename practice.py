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

for i in range(1, 6):
    x = range(1, len(data_dict[i])+1)
    plt.plot(x, data_dict[i])
plt.show()
