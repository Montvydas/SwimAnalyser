import pylab
import numpy as np
from matplotlib import pyplot as plt
import analyser
from peak_detector import detect_peaks, detect_peaks_from_maximums, peakdet


def plot_significant_peaks(y):
    lag = 500
    threshold = 4.5
    influence = 0

    result = detect_peaks(y, lag=lag, threshold=threshold, influence=influence)
    plt.figure(figsize=(20, 10))

    pylab.subplot(211)
    pylab.plot(np.arange(1, len(y) + 1), y)
    pylab.plot(np.arange(1, len(y)+1), result["avgFilter"], color="cyan", lw=2)
    pylab.plot(np.arange(1, len(y)+1), result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    pylab.plot(np.arange(1, len(y)+1), result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    pylab.subplot(212)
    pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    pylab.ylim(-1.5, 1.5)
    pylab.grid()
    pylab.show()


def plot_lag_info(y, lag, title):
    result = analyser.get_lag_info(y, lag)
    plt.figure(figsize=(20, 10))
    pylab.plot(np.arange(1, len(y) + 1), y, label="original")

    pylab.plot(np.arange(1, len(y) + 1), result["avgFilter"], color="black", lw=1, label="Average")
    pylab.plot(np.arange(1, len(y) + 1), result["stdFilter"], color="green", lw=1, label="STD")
    pylab.plot(np.arange(1, len(y) + 1), result["medFilter"], color="red", lw=1, label="Median")
    pylab.title(title)
    pylab.legend()
    pylab.grid()
    pylab.show()



def plot_peaks_from_max(y, threshold):
    lag = 500
    # 1/2 for pro swimmer, 1/4 for simple, 2/3 for lap count
    # threshold = 2 / 3

    data = [x ** 2 for x in y]
    result = detect_peaks_from_maximums(data, lag=lag, threshold=threshold)

    pylab.plot(np.arange(1, len(y) + 1), data)
    pylab.plot(np.arange(1, len(y) + 1), [0] * lag + result)
    pylab.show()


def plot_all_data(data):
    # plt.figure(figsize=(20, 10))
    plt.plot(data[0], 'r-', linewidth=2, label='pitch', alpha=0.7)
    plt.plot(data[1], 'g-', linewidth=2, label='roll', alpha=0.7)
    plt.plot(data[2], 'b-', linewidth=2, label='yaw', alpha=0.7)

    plt.grid()
    plt.legend()

    plt.show()

def plot_single(data, legend):
    # plt.figure(figsize=(20, 10))
    plt.plot(data, 'b-', linewidth=2, label=legend, alpha=0.7)
    plt.grid()
    plt.legend()
    plt.show()


def plot_peakdet(y, threshold=50):
    maxtab, mintab = peakdet(y, threshold)

    print ('In Total Max peaks=', len(maxtab), ' min peaks=', len(mintab))

    # plt.figure(figsize=(20, 10))
    plt.plot(y, 'g-', linewidth=2, label='yaw', alpha=0.7)
    plt.scatter(maxtab[:, 0], maxtab[:, 1], color='blue')
    plt.scatter(mintab[:, 0], mintab[:, 1], color='red')

    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.show()
