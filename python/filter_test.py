import numpy as np
import argparse
from matplotlib import pyplot as plt
import math
import sys

import analyser
import plot_results

# https://www.codeproject.com/Tips/1092012/A-Butterworth-Filter-in-Csharp
# https://dsp.stackexchange.com/questions/9360/where-can-i-find-a-table-of-z-domain-coefficients-for-butterworth-filters


class Filter:
    def __init__(self, cutoff, fs):
        self.v = [0, 0, 0]
        f = math.tan(math.pi * cutoff / fs)
        f2 = f * f
        sq2 = math.sqrt(2)
        a0 = 1 + sq2 * f + f2

        # Coefficients for 2nd order buttorworth filter
        self.a1 = -2 * (f2 - 1) / a0
        self.a2 = -(1 - sq2 * f + f2) / a0
        self.a0 = f2 / a0
        print (self.a0, " ", self.a1, " ", self.a2)

    def step(self, x):
        self.v[0] = self.v[1]
        self.v[1] = self.v[2]
        tmp = (self.a0 * x) + (self.a2*self.v[0])+(self.a1*self.v[1])
        self.v[2] = tmp
        return (self.v[0] + self.v[2])+2 * self.v[1]

if __name__ == "__main__":
    # f = math.tan(math.pi * 0.05 / 1)
    # f2 = f*f
    # sq2 = math.sqrt(2)
    # a0 = 1 + sq2 * f + f2
    # a1 = -2 * (f2 - 1) / a0
    # a2 = -(1 - sq2 * f + f2) / a0
    # a0 = a0
    #
    # print (a0, " ", a1, " ", a2)
    #
    # wc = math.tan(0.05 * math.pi / 1)
    # k1 = 1.414213562 * wc
    # k2 = wc * wc
    # a = k2 / (1 + k1 + k2)
    # b = 2 * a
    # c = a
    # k3 = b / k2
    # d = -2 * a + k3
    # e = 1 - (2 * a) - k3
    #
    # print (a, " ", b, " ", c, " ", d, " ", e)
    #
    # f = math.tan(math.pi * 0.05 / 1)
    # f2 = f*f
    # sq2 = math.sqrt(2)
    # a0 = 1 + sq2 * f + f2
    #
    # a = f2 / a0
    # b = 1 - (2 * f2 / a0) - 2 / a0
    # c = -2 * (f2 - 1) / a0
    #
    # print (a, " ",  b, " ", c)

    # sys.exit()

    data = [0]*100 + [10]*100 #+ [5]*20 + [100]*30 + [0]*50

    # data = [math.sqrt(x) for x in range(100)]

    # data = [x+np.random.normal(loc=0.0, scale=2.0) for x in data]

    # data = data[::2]

    filtered = analyser.butter_lowpass_filter(data,0.05,1,2)
    # filtered = analyser.ellip_lowpass_filter(data,0.02,1,3)
    filter = Filter(0.05, 1)
    filtered2 = []
    for d in data:
        filtered2.append(filter.step(d))


    # plot_results.plot_single(data, "original")
    plot_results.plot_all_data([data, filtered, filtered2])