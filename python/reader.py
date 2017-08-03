import csv
import numpy as np


def read_csv(name):
    file_csv = open(name, 'rt')
    reader_csv = csv.reader(file_csv)
    raw_data_csv = [x for x in reader_csv]
    # data_csv = np.array(raw_data_csv)
    data_csv = [map(float, x) for x in raw_data_csv[1:]]
    file_csv.close()
    return [i for i in zip(*data_csv)]
    # return data_csv


def read_bin(name):
    bytes_lst = []
    with open(name, "rb") as f:
        byte = f.read(1)
        while byte:
            bytes_lst.append(byte)
            byte = f.read(1)
    return bytes_lst


def read_data_from_bytes(bytes_lst, skip_count, sensitivity, type):
    total = int(len(bytes_lst) / 3)
    print (total / 2 - skip_count)

    rawX = []
    rawY = []
    rawZ = []

    skip = 0
    for i in range(0, total, 2):
        skip += 1
        if skip < skip_count:
            continue


        x = int.from_bytes(bytes_lst[i] + bytes_lst[i + 1], byteorder='little', signed=True)
        x /= sensitivity
        rawX.append(x)

        y = int.from_bytes(bytes_lst[total + i] + bytes_lst[total + i + 1], byteorder='little', signed=True)
        y /= sensitivity
        rawY.append(y)

        z = int.from_bytes(bytes_lst[2 * total + i] + bytes_lst[2 * total + i + 1], byteorder='little', signed=True)
        z /= sensitivity
        rawZ.append(z)

    # TODO find out if this is correct!!!
    if type == 'gyro':
        return [rawX, rawY, rawZ]
    elif type == 'acc':
        return [rawX, rawY, rawZ]
    elif type == 'mag':
        return [rawX, rawY, rawZ]
