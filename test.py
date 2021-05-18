import matplotlib.pyplot as plt

for i in range(1, 6):
    with open("test{}.txt".format(i), "r") as f:
        y = f.readlines()
        x = len(y) + 1
        plt.plot(x, y)
plt.legend(('1st', '2nd', '3rd', '4th', '5th'))
plt.show()
