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

from analyser import low_pass_filter, get_angles_from_acc


'''
Ideas:


pitch = detrend(pitch_acc) # Is used to detrend a line


# Filter requirements.
order = 6
fs = 48.0      # sample rate, Hz
cutoff = .1  # desired cutoff frequency of the filter, Hz
yaw = butter_lowpass_filter(yaw, 0.08, fs, order) # Apply a filter


yaw = [x/20.0 for x in yaw]   # yaw might need to be adjusted


'''


def count_laps_undersampled(data, running_mean_number, cutoff, fs, order, divider, threshold):
    print ("actual cutoff=", 2 * cutoff / fs)

    # Firstly lower the sample rate
    data = data[::divider]
    plt.plot(data, 'r-', linewidth=2, label="divider=" + str(divider), alpha=0.7)

    # Apply running mean
    data = analyser.runningMeanFast(data, running_mean_number)
    plt.plot(data, 'g-', linewidth=2, label="moving average", alpha=0.7)

    # Apply filer
    data = analyser.butter_lowpass_filter(data, cutoff, fs, order)
    plot_results.plot_single(data, "filtered")

    # get differential
    diff = np.gradient(data)
    diff = [x ** 2 for x in diff]

    # Plot the result and try tp find peaks
    plot_results.plot_peakdet(diff, threshold)


def count_laps_butter_only(data, cutoff, fs, order, threshold):
    print ("actual cutoff=", 2 * cutoff / fs)
    plt.plot(data, 'g-', linewidth=2, label="original", alpha=0.7)

    # Add a butter filter to remove  variations
    data = analyser.butter_lowpass_filter(data, cutoff, fs, order)
    # data = analyser.cheby_lowpass_filter(data, cutoff, fs, order)
    plot_results.plot_single(data, "filtered")

    diff = np.gradient(data)
    diff = [x ** 2 for x in diff]
    plot_results.plot_peakdet(diff, threshold)


def count_laps(data, running_mean_number, cutoff, fs, order, threshold):
    print ("actual cutoff=", 2 * cutoff / fs)
    plt.plot(data, 'g-', linewidth=2, label="original", alpha=0.7)
    # plot_results.plot_single(data, "original")

    data = analyser.runningMeanFast(data, running_mean_number)
    # plot_results.plot_lag_info(data, 400)

    plt.plot(data, 'r-', linewidth=2, label="moving average", alpha=0.7)
    data = analyser.butter_lowpass_filter(data, cutoff, fs, order)
    plot_results.plot_single(data, "filtered")

    apply_all_lag_analysis(data, "FilteredYaw")


    # plot_results.plot_lag_info(data, 300)

    # data = [x**2 for x in data]
    # plot_results.plot_significant_peaks(data)

    diff = np.gradient(data)
    diff = [x ** 2 for x in diff]
    plot_results.plot_peakdet(diff, threshold)

    return data


def apply_all_lag_analysis(data, title):
    plot_results.plot_lag_info(data, 50, title + ", lag=50")
    plot_results.plot_lag_info(data, 200, title + ", lag=200")
    plot_results.plot_lag_info(data, 500, title + ", lag=500")
    plot_results.plot_lag_info(data, 1000, title + ", lag=1000")

# Raw acceleration can be used to detect flips
# applied transformation on yaw can also detect flips

# Strokes can be counted using simple roll from accelerometer

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
    parser.add_argument('accelerometer', type=str, help='filename of data file to be read')
    parser.add_argument('gyroscope', type=str, help='filename of data file to be read')
    parser.add_argument('magnetometer', type=str, help='filename of data file to be read')
    args = parser.parse_args()

    a = reader.read_bin(args.accelerometer)
    g = reader.read_bin(args.gyroscope)
    m = reader.read_bin(args.magnetometer)

    a = reader.read_data_from_bytes(a, SKIP_COUNT, ACC_SENSITIVITY, 'acc')
    g = reader.read_data_from_bytes(g, SKIP_COUNT, GYRO_SENSITIVITY, 'gyro')
    m = reader.read_data_from_bytes(m, SKIP_COUNT, GYRO_SENSITIVITY, 'mag')

    total = len(a[0])

    order = 2
    fs = FREQUENCY # sample rate, Hz
    cutoff = 10  # desired cutoff frequency of the filter, Hz

    print ("actual cutoff=", 2 * cutoff / fs)
    # a = analyser.apply_butter_low_pass_to_all(a, cutoff, fs, order)
    # g = analyser.apply_butter_low_pass_to_all(g, cutoff, fs, order)
    # m = analyser.apply_butter_low_pass_to_all(m, cutoff, fs, order)

    # a_angles = analyser.get_angles_from_acc(a)
    # g_angles = analyser.get_angles_from_gyro(g, 0.01)
    # angles = analyser.get_angles_from_combined(a_angles, g, 0.01, 0.1)

    # plot_results.plot_all_data(a_angles)
    # plot_results.plot_all_data(g_angles)
    # plot_results.plot_all_data(angles)
    # sys.exit()

    # plot_results.plot_all_data(g)

    # plot_results.plot_all_data(a)

    # NO
    # tmp = a[1]
    # a[1] = a[2]
    # a[2] = tmp

    # NO
    # tmp=a[0]
    # a[0]=a[1]
    # a[1]=a[2]
    # a[2]=tmp

    # Works well for dummy freestyle, but not for test2, test1 is OK. Why?
    # tmp = a[0]
    # a[0]=a[1]
    # a[1]=tmp

    # Works when used for dummy-freestyle, but doesn't work for test1 and test2
    # tmp = a[0]
    # a[0] = a[2]
    # a[2] = a[1]
    # a[1] = tmp

    # Works very well for yaw!!!! Ask maybe gyro uses a different system?
    # tmp = a[0]
    # a[0] = a[2]
    # a[2] = tmp

    # sys.exit()

    # all_a = [math.sqrt(a[0][i] ** 2 + a[1][i] ** 2 + a[2][i] ** 2) for i in range(total)]

    time = [i/FREQUENCY for i in range(total+1)]

    # angles6DOF = analyser.PerformMadgwickQuaternion6DOFOriginal(a, g, time)
    angles6DOF = analyser.PerformMadgwickQuaternion6DOF(a, g, time)
    angles9DOF = analyser.PerformMadgwickQuaternion9DOF(a, g, m, time)

    # plot_results.plot_single(angles6DOF[2], "yaw")
    pitch6DOF = analyser.to_continuous(angles6DOF[0])
    roll6DOF = analyser.to_continuous(angles6DOF[1])
    yaw6DOF = analyser.to_continuous(angles6DOF[2])

    # Remove jumps from -180 to 180

    plot_results.plot_all_data([pitch6DOF, roll6DOF, yaw6DOF])

    # plot_results.plot_single(angles6DOF[1], "roll")
    # sys.exit()

    # apply_all_lag_analysis(a[0], "Acc[0]")

    # apply_all_lag_analysis(a[1], "Acc[1]")
    # apply_all_lag_analysis(a[2], "Acc[2]")
    #
    # apply_all_lag_analysis(g[0], "Gyro[0]")
    # apply_all_lag_analysis(g[0], "Gyro[1]")
    # apply_all_lag_analysis(g[0], "Gyro[2]")
    #
    # apply_all_lag_analysis(yaw6DOF, "Yaw")
    # apply_all_lag_analysis(angles6DOF[1], "Roll")
    # apply_all_lag_analysis(pitch6DOF, "Pitch")

    # sys.exit()

    # Angle lag analysis
    # plot_results.plot_lag_info(yaw6DOF, 400)

    # sys.exit()

    # To save data for exportation
    # analyser.save_to_file("quaternion_9dof.txt", angles6DOF)
    # scipy.io.savemat('arrdata.mat', mdict={'arr': angles6DOF[2]})

    order = 2
    fs = FREQUENCY
    cutoff = 0.1
    # count_laps_butter_only(data=yaw6DOF, cutoff=cutoff, fs=FREQUENCY, order=3, threshold=0.1)

    order = 2
    fs = FREQUENCY
    cutoff = 0.08
    # bias = count_laps(data=yaw6DOF, running_mean_number=500, cutoff=cutoff, fs=FREQUENCY, order=2, threshold=0.04)
    # nobias = [yaw6DOF[i] - bias[i] for i in range(len(yaw6DOF))]
    # plot_results.plot_single(nobias, "unbiased")

    cutoff = 10
    fs = 48
    # count_laps_undersampled(data=yaw6DOF, running_mean_number=5, cutoff=10, fs=48, order=2, divider=100, threshold=600)
