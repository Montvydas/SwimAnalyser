import numpy as np
import argparse
import sys

from scipy.signal import butter, lfilter, freqz, find_peaks_cwt, detrend
from matplotlib import pyplot as plt
from peak_detector import detect_peaks, detect_peaks_from_maximums, peakdet
import math

import scipy.signal as scisignal
import scipy.io

import reader
import analyser
import plot_results


# ACC_SENSITIVITY = 9.81  # Not used for angle calculations but useful when analysing used energy etc.
ACC_SENSITIVITY = 1000  # Not used for angle calculations but useful when analysing used energy etc.
# GYRO_SENSITIVITY = 939.7   # e.g. if rotated 360 degrees, this scale adjusts the result to be 360 degrees
GYRO_SENSITIVITY = 1000   # e.g. if rotated 360 degrees, this scale adjusts the result to be 360 degrees
MAG_SENSITIVITY = 1000   # e.g. if rotated 360 degrees, this scale adjusts the result to be 360 degrees

FREQUENCY = 100      # Frequency in hertz of the update rate
dT = 1/FREQUENCY   # Update rate in seconds

SKIP_COUNT = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('all_data', type=str, help='filename of data file to be read')
    args = parser.parse_args()

    all_data = reader.read_csv(args.all_data)
    time = all_data[0]
    a = all_data[1:4]
    g = all_data[4:]

    order = 2
    fs = FREQUENCY # sample rate, Hz
    cutoff = 10  # desired cutoff frequency of the filter, Hz

    print ("actual cutoff=", 2 * cutoff / fs)
    a = analyser.apply_butter_low_pass_to_all(a, cutoff, fs, order)
    g = analyser.apply_butter_low_pass_to_all(g, cutoff, fs, order)

    a[0] = [x*2.0*9.8 / 32768.0 for x in a[0]]
    a[1] = [x*2.0*9.8 / 32768.0 for x in a[1]]
    a[2] = [x*2.0*9.8 / 32768.0 for x in a[2]]

    total = len(a[0])
    # time = [i / FREQUENCY for i in range(total + 1)]
    time = [x / 1000000.0 for x in time]
    time.insert(0, time[0]-100)

    g[0] = [(x)*250.0*math.pi/180.0/32768.0 for x in g[0]]
    g[1] = [(x)*250.0*math.pi/180.0/32768.0 for x in g[1]]
    g[2] = [(x)*250.0*math.pi/180.0/32768.0 for x in g[2]]

    a_total = [math.sqrt(a[0][i]**2 + a[1][i]**2 + a[2][i]**2) for i in range(total)]

    # plot_results.plot_all_data([[0]*total, [0]*total, a_total])
    plot_results.plot_all_data(a)
    plot_results.plot_all_data(g)

    a_angles = analyser.get_angles_from_acc(a)
    a_angles = analyser.to_degrees(a_angles)

    g_angles = analyser.get_angles_from_gyro(g, 0.01)

    # plot_results.plot_all_data(a_angles)
    # plot_results.plot_all_data(g_angles)

    # sys.exit()
    # angles6DOF = analyser.PerformMadgwickQuaternion6DOFOriginal(a, g, time)
    angles6DOF = analyser.PerformMadgwickQuaternion6DOF(a, g, time)

    # plot_results.plot_single(angles6DOF[2], "yaw")
    pitch6DOF = analyser.to_continuous(angles6DOF[0])
    roll6DOF = analyser.to_continuous(angles6DOF[1])
    yaw6DOF = analyser.to_continuous(angles6DOF[2])

    # Remove jumps from -180 to 180

    plot_results.plot_all_data([pitch6DOF, roll6DOF, yaw6DOF])