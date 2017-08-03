import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz, find_peaks_cwt, detrend
import reader
import plot_results
import analyser
import math
import peak_detector

ACC_SENSITIVITY = 9.81  # Not used for angle calculations but useful when analysing used energy etc.
GYRO_SENSITIVITY = 939.7   # e.g. if rotated 360 degrees, this scale adjusts the result to be 360 degrees

FREQUENCY = 20.0      # Frequency in hertz of the update rate
dT = 1.0/FREQUENCY   # Update rate in seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('accelerometer', type=str, help='filename of data file to be read')
    parser.add_argument('gyroscope', type=str, help='filename of data file to be read')
    parser.add_argument('magnetometer', type=str, help='filename of data file to be read')
    parser.add_argument('--data_type', type=str, help='csv or bin')
    parser.add_argument('--check', type=bool, help='checks user input')
    args = parser.parse_args()

    a = reader.read_csv(args.accelerometer)
    g = reader.read_csv(args.gyroscope)
    m = reader.read_csv(args.magnetometer)

    a_time = list(a[0])
    a_time.insert(0, 0)
    a = a[1:]
    g_time = list(g[0])
    g_time.insert(0, 0)
    g = g[1:]
    m_time = list(m[0])
    m_time.insert(0, 0)
    m = m[1:]

    time = [x/1000.0 for x in g_time]

    total = len(a[0])

    # plot_results.plot_all_data(g)

    # plot_results.plot_all_data(a)
    # plot_results.plot_all_data(g)
    # plot_results.plot_all_data(m)

    # all_a = [a[0][i]**2 + a[1][i]**2 + a[2][i]**2 for i in range(total)]

    # order = 2
    # fs = 48.0  # sample rate, Hz
    # cutoff = 5  # desired cutoff frequency of the filter, Hz
    #
    # a = analyser.apply_butter_low_pass_to_all(a, cutoff, fs, order)
    # g = analyser.apply_butter_low_pass_to_all(g, cutoff, fs, order)
    # m = analyser.apply_butter_low_pass_to_all(m, cutoff, fs, order)

    # all_a_filtered = analyser.butter_lowpass_filter(all_a, cutoff, fs, order)


    ### This one trying to figure out if swimming or not swimming!!
    # samples = 100
    # std = [np.std(all_a[:samples])] * samples
    # std = [0]*samples
    # std_filtered = [np.std(all_a_filtered[:samples])] * samples
    # std_filtered = [0] * samples
    #
    # amplitudes = [0] * samples
    # means = [0] * samples
    #
    # swimming = False
    # swimming_time = []
    #
    # for i in range(samples, total):
    #     mstd = np.std(all_a[i-samples:i])
    #     std.append(mstd)
    #     mstd_filtered = np.std(all_a_filtered[i-samples:i])
    #     std_filtered.append(mstd_filtered)
    #
    #     maxi = max(all_a_filtered[i-samples:i])
    #     mini = min(all_a_filtered[i-samples:i])
    #     amp = maxi - mini
    #     amplitudes.append(amp)
    #
    #     if amp > 80 and mstd_filtered > 20 and not swimming:
    #         # swimming = True
    #         swimming_time.append(i)

    # a_angles = analyser.get_angles_from_acc(a)
    # g_angles = analyser.get_angles_from_gyro(g, dT)
    # combined_angles = analyser.get_angles_from_combined(a_angles, g, dT, 0.9)


    # plt.plot(swimming_time, [10]*len(swimming_time), 'ro')
    # plot_results.plot_all_data(a)

    # a_angles_deg = analyser.to_degrees(a_angles)
    # g_angles_deg = analyser.to_degrees(g_angles)
    # combined_angles_deg = analyser.to_degrees(combined_angles)

    # plot_results.plot_peakdet(a[3], 3)
    # plot_results.plot_peakdet(a_angles[0], 3)

    # plot_results.plot_all_data(a_angles_deg)
    # plot_results.plot_all_data(g_angles_deg)
    # plot_results.plot_all_data(combined_angles_deg)
    # plot_results.plot_all_data(a)
    # plot_results.plot_all_data(g)

    # tmp = a[0]
    # a[0] = a[2]
    # a[2] = tmp

    # angles6DOF = analyser.PerformMadgwickQuaternion6DOFOriginal(a, g, time)
    angles6DOF = analyser.PerformMahonyQuaternion6DOF(a, g, time)
    angles9DOF = analyser.PerformMadgwickQuaternion9DOF(a, g, m, time)

    plot_results.plot_all_data(a)
    plot_results.plot_all_data(g)

    order = 2
    fs = 48.0  # sample rate, Hz
    cutoff = 0.5  # desired cutoff frequency of the filter, Hz

    # swim_after_setup = [x for x in swimming_time if x > 200]

    # plot_results.plot_all_data(a)
    # plot_results.plot_all_data(g)
    # plot_results.plot_all_data(angles6DOF)

    angles6DOF[2] = analyser.to_continuous(angles6DOF[2])

    import scipy.io
    # scipy.io.savemat('arrdata2.mat', mdict={'arr2': angles6DOF[2]})

    #
    plt.plot(angles6DOF[2], 'g-', linewidth=2, label="original", alpha=0.7)
    y_av = analyser.runningMeanFast(angles6DOF[2], 20)

    # y_av = analyser.low_pass_filter(angles6DOF[2], 0.1)

    plt.plot(y_av, 'r-', linewidth=2, label="smoothing filter", alpha=0.7)

    y_av = analyser.butter_lowpass_filter(y_av, 1, 48, 2)
    plot_results.plot_single(y_av, "filtered")

    from numpy import diff
    # dx = 1
    diff = np.gradient(y_av)
    # diff = diff(y_av) / dx
    diff = [x ** 2 for x in diff]
    plot_results.plot_peakdet(diff, 15)

    import sys
    sys.exit()

    # variance = np.var(angles6DOF[2])
    angles6DOF[2] = analyser.butter_lowpass_filter(angles6DOF[2], cutoff, fs, order)

    # plt.plot(swim_after_setup, [10] * len(swim_after_setup), 'ro')
    diff = np.gradient(angles6DOF[2])
    diff = [math.fabs(x) for x in diff]
    plot_results.plot_peakdet(diff, 3)

    # plot_results.plot_all_data([amplitudes, std_filtered, angles6DOF[2]])

    angles9DOF[2] = analyser.to_continuous(angles9DOF[2])
    # angles9DOF[2] = analyser.butter_lowpass_filter(angles9DOF[2], cutoff, fs, order)

    # plot_results.plot_all_data(angles6DOF)
    plot_results.plot_single(diff, "diff")

    analyser.save_to_file("quaternion_9dof.txt", angles6DOF)

    # This value is good to detect sqrt(x**2 + y**2 + z**2) of all acceleration
    # plot_results.plot_peakdet(a_angles[2], 7)
    # plt.show()
