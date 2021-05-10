import matplotlib.pyplot as plt

data_dict = {}
for i in range(1, 6):
    with open("test{}.txt".format(i), "r") as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            line = line.strip("[").strip().strip("]")
            if line:
                data_list.extend([float(i) for i in line.split()])
        data_dict[i] = data_list

for i in range(1, 6):
    x = range(1, len(data_dict[i])+1)
    plt.plot(x, data_dict[i])
plt.legend(('1st', '2nd', '3rd', '4th', '5th'))
plt.show()
