#! -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt

x = []
y1 = []
y2 = []
with open('log', 'r') as f:
    reader = csv.reader(f)
    # header = next(reader)
    for row in reader:
        x.append(int(row[0]))
        y1.append(float(row[1]))
plt.plot(x, y1, 'or', label="random")
plt.xlabel("minute")
plt.ylabel("best score")
plt.legend(loc="lower right")
plt.savefig("score.png")
