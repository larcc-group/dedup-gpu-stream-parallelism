#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

with open("rabin_fingerprint_size_large.txt","r") as file:
    lines = [int(x) for x in file.readlines()]


plt.figure()
n,bins,patches= plt.hist(lines, density=True, bins=50)
print(bins)
for bina in zip(bins,n):
    print("até {0}: {1:.2f}".format(*bina))
plt.show()