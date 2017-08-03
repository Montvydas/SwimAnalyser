from scipy.signal import butter, lfilter, cheby1, ellip
import math
from math import cos, sin, sqrt, atan2, asin, atan
import numpy as np


def low_pass_filter(data, coeff):
    output = []
    prev = data[0]
    for d in data:
        prev = coeff * d + prev * (1 - coeff)
        output.append(prev)
    return output


def apply_low_pass_to_all(data, coeff):
    for i in range(len(data)):
        data[i] = low_pass_filter(data[i], coeff)
    return data

# Example is given:
# yaw = butter_lowpass_filter(yaw, 0.08, fs, order)
# accX = butter_lowpass_filter(accX, cutoff, fs, order)
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print("testing normal cutoff is", normal_cutoff)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def ellip_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print("testing normal cutoff is", normal_cutoff)
    b, a = ellip(order, 2, 20, normal_cutoff, btype='low', analog=False)
    return b, a

def ellip_lowpass_filter(data, cutoff, fs, order=5):
    b, a = ellip_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_butter_low_pass_to_all(data, cutoff, fs, order=5):
    for i in range(len(data)):
        data[i] = butter_lowpass_filter(data[i], cutoff, fs, order)
    return data

# returns angles in degrees and total gravity in default units
def get_angles_from_acc(acc):
    pitch = []
    roll = []
    gravity = []

    for i in range(len(acc[0])):
        # pitch.append(math.atan2(acc[0][i], acc[1][i]) * 57.3)

        pitch.append(math.atan2(acc[1][i], math.sqrt(acc[0][i] ** 2 + acc[2][i] ** 2)))
        roll.append(math.atan2(-acc[0][i], acc[2][i]))
        gravity.append(math.sqrt(acc[0][i] ** 2 + acc[1][i] ** 2 + acc[2][i] ** 2))

        # A different setting, the top one worked together with gyro
        # roll.append(atan2(acc[1][i], acc[2][i]) * 57.3)
        # pitch.append(atan2(-acc[0][i], sqrt(acc[1][i] ** 2 + acc[2][i] ** 2)) * 57.3)


        # pitch.append(math.atan2(accX[i], accY[i]) * 57.3)

        # These are x, y, z values
        # pitch.append(math.atan2(accY[i], math.sqrt(accX[i] ** 2 + accZ[i] ** 2)) - PITCH_OFFSET)
        # roll.append(math.atan2(-accX[i], accZ[i]) - ROLL_OFFSET)
        # gravity.append(math.sqrt(accX[i] ** 2 + accY[i] ** 2 + accZ[i] ** 2))

        # pitch.append(math.asin(accX[i] / (math.sqrt(accX[i] ** 2 + accY[i] ** 2 + accZ[i] ** 2))) * 57.3)
        # pitch.append(math.asin(acc[0][i] / (math.sqrt(acc[0][i] ** 2 + acc[1][i] ** 2 + acc[2][i] ** 2))) * 57.3)
    return [pitch, roll, gravity]


# returns angles in degrees
def get_angles_from_gyro(gyro, dT):
    pitch = []
    roll = []
    yaw = []

    p = 0.0
    r = 0.0
    y = 0.0

    for i in range(len(gyro[0])):
        p += gyro[0][i] * dT
        r += gyro[1][i] * dT
        y += gyro[2][i] * dT

        # p -= gyroZ * dT
        # r += gyroY * dT
        # y += gyroX * dT

        pitch.append(p)
        roll.append(r)
        yaw.append(y)

    return [pitch, roll, yaw]


# All passed values will be translated to degrees
def to_degrees(data):
    ret = []
    for sub in data:
        ret.append([x * 57.3 for x in sub])
    return ret


# Here angles are already in degrees
def get_angles_from_combined(acc_angles, gyro, dT, alpha=0.0):
    pitch = []
    roll = []
    yaw = []

    p = 0.0
    r = 0.0
    y = 0.0

    for i in range(len(acc_angles[0])):
        # p = alpha * (p - pitch_gyro_raw[i] * dT) + (1 - alpha) * pitch_acc[i]
        # r = alpha * (r + roll_gyro_raw[i] * dT) + (1 - alpha) * roll_acc[i]
        # y += yaw_gyro_raw[i] * dT

        p = alpha * (p + gyro[0][i] * dT) + (1 - alpha) * acc_angles[0][i]
        r = alpha * (r + gyro[1][i] * dT) + (1 - alpha) * acc_angles[1][i]
        y += gyro[2][i] * dT

        pitch.append(p)
        roll.append(r)
        yaw.append(y)

    return [pitch, roll, yaw]


def to_world_coordinate_system_1(angles):
    pitch = []
    roll = []
    yaw = []
    for i in range(len(angles[0])):
        p = angles[0][i]
        r = angles[1][i]
        y = angles[2][i]
        # yaw, then pitch,then roll
        ox0 = cos(y) * cos(p) + \
            (cos(y) * sin(p) * sin(r) - sin(y) * cos(r)) + \
            (cos(y) * sin(p) * cos(r) + sin(y) * sin(r))
        oy0 = sin(y) * cos(p) + \
            (sin(y) * sin(p) * sin(r) + cos(y) * cos(r)) + \
            (sin(y) * sin(p) * cos(r) - cos(p) * sin(r))
        oz0 = -sin(r) + \
            cos(p) * sin(r) + \
            cos(p) * cos(r)
        pitch.append(ox0)
        roll.append(oy0)
        yaw.append(oz0)
    return [pitch, roll, yaw]


# found this to work better?
def to_world_coordinate_system_2(angles):
    pitch = []
    roll = []
    yaw = []
    for i in range(len(angles[0])):
        p = angles[0][i]
        r = angles[1][i]
        y = angles[2][i]

        ox0 = -cos(y)*sin(p)*sin(r) - sin(y)*cos(r)
        oy0 = -sin(y)*sin(p)*sin(r) + cos(y)*cos(r)
        oz0 = cos(p)*sin(r)

        pitch.append(ox0)
        roll.append(oy0)
        yaw.append(oz0)
    return [pitch, roll, yaw]


def print_significant_values(data):
    print ("average is", np.mean(data))
    print ("std is", np.std(data))
    print ("median is", np.median(data))


def save_to_file(name, data):
    text_file = open(name, "w")
    text_file.writelines([str(int(data[0][i])) + " "
                          + str(int(data[1][i])) + " "
                          + str(int(data[2][i])) + "\n"
                          for i in range(len(data[0]))])
    text_file.close()

# Fast version of inverse square
# def invSqrt(x):
# 	halfx = 0.5 * x
# 	y = x
# 	i = *(long*)&y
# 	i = 0x5f3759df - (i>>1);
# 	y = *(float*)&i;
# 	y = y * (1.5f - (halfx * y * y));
# 	return y


# Found optimal case for our data
twoKpDef = (2.0 * 0.3)	# 2 * proportional gain
twoKiDef = (2.0 * 0.2)	# 2 * integral gain
twoKp = twoKpDef
twoKi = twoKiDef


def MahonyAHRSupdateIMU(ax, ay, az, gx, gy, gz, q, deltat, integralFB):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    integralFBx = integralFB[0]
    integralFBy = integralFB[1]
    integralFBz = integralFB[2]
# // Normalise accelerometer measurement
    # // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    # if not (ax == 0.0 and ay == 0.0 and az == 0.0):
        # // Normalise Normaliseaccelerometer measurement
    norm = sqrt(ax * ax + ay * ay + az * az)
    # // handle NaN
    if norm == 0.0: return
    norm = 1.0 / norm
    
# norm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= norm
    ay *= norm
    az *= norm

    # // Estimated direction of gravity and vector perpendicular to magnetic flux
    halfvx = q1 * q3 - q0 * q2
    halfvy = q0 * q1 + q2 * q3
    halfvz = q0 * q0 - 0.5 + q3 * q3

    # // Error is sum of cross product between estimated and measured direction of gravity
    halfex = (ay * halfvz - az * halfvy);
    halfey = (az * halfvx - ax * halfvz);
    halfez = (ax * halfvy - ay * halfvx);

    # // Compute and apply integral feedback if enabled
    if twoKi > 0.0:
        integralFBx += twoKi * halfex * deltat # // integral error scaled by Ki
        integralFBy += twoKi * halfey * deltat
        integralFBz += twoKi * halfez * deltat
        gx += integralFBx #// apply integral feedback
        gy += integralFBy
        gz += integralFBz
    else:
        integralFBx = 0.0 #// prevent integral windup
        integralFBy = 0.0
        integralFBz = 0.0

    # // Apply proportional feedback
    gx += twoKp * halfex
    gy += twoKp * halfey
    gz += twoKp * halfez

    # // Integrate rate of change of quaternion
    gx *= (0.5 * deltat) #// pre-multiply common factors
    gy *= (0.5 * deltat)
    gz *= (0.5 * deltat)
    qa = q0
    qb = q1
    qc = q2
    q0 += (-qb * gx - qc * gy - q3 * gz)
    q1 += (qa * gx + qc * gz - q3 * gy)
    q2 += (qa * gy - qb * gz + q3 * gx)
    q3 += (qa * gz + qb * gy - qc * gx)

    # // Normalise quaternion
    # norm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)  # // normalise quaternion
    norm = 1.0 / norm
    q0 *= norm
    q1 *= norm
    q2 *= norm
    q3 *= norm
    return [q0, q1, q2, q3], [integralFBx, integralFBy, integralFBz]


def MadgwickAHRSupdateIMU(ax, ay, az, gx, gy, gz, q, deltat, beta):
    # float norm;
    # float s0, s1, s2, s3;
    # float qDot1, qDot2, qDot3, qDot4;
    # float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # Rate of change of quaternion from gyroscope
    qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
    qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
    qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
    qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

    # // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    # if not (ax == 0.0 and ay == 0.0 and az == 0.0):
        # // Normalise Normaliseaccelerometer measurement
    norm = sqrt(ax * ax + ay * ay + az * az)
    # // handle NaN
    if norm == 0.0: return
    norm = 1.0 / norm

    # norm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= norm
    ay *= norm
    az *= norm

    # // Auxiliary variables to avoid repeated arithmetic
    _2q0 = 2.0 * q0
    _2q1 = 2.0 * q1
    _2q2 = 2.0 * q2
    _2q3 = 2.0 * q3
    _4q0 = 4.0 * q0
    _4q1 = 4.0 * q1
    _4q2 = 4.0 * q2
    _8q1 = 8.0 * q1
    _8q2 = 8.0 * q2
    q0q0 = q0 * q0
    q1q1 = q1 * q1
    q2q2 = q2 * q2
    q3q3 = q3 * q3

    # // Gradient decent algorithm corrective step
    s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
    s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
    s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
    s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

    norm = sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)  # // normalise step magnitude
    norm = 1.0 / norm

    # norm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3) # // normalise step magnitude
    s0 *= norm
    s1 *= norm
    s2 *= norm
    s3 *= norm

    # // Apply feedback step
    qDot1 -= beta * s0
    qDot2 -= beta * s1
    qDot3 -= beta * s2
    qDot4 -= beta * s3

    # // Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * deltat
    q1 += qDot2 * deltat
    q2 += qDot3 * deltat
    q3 += qDot4 * deltat

    # // Normalise quaternion
    norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)  # // normalise quaternion
    norm = 1.0 / norm
    # norm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= norm
    q1 *= norm
    q2 *= norm
    q3 *= norm

    return [q0, q1, q2, q3]

GyroMeasError = math.pi * (40.0 / 180.0)    # gyroscope measurement error in rads/s (start at 60 deg/s), then reduce after ~10 s to 3
# beta = sqrt(3.0 / 4.0) * GyroMeasError      # compute beta = 0.605
GyroMeasDrift = math.pi * (2.0 / 180.0)     # gyroscope measurement drift in rad/s/s (start at 0.0 deg/s/s)
# zeta = sqrt(3.0 / 4.0) * GyroMeasDrift      # compute zeta, the other free parameter in the Madgwick scheme usually set to a small or zero value = 0.03
beta = 0.041                              # After ~10 seconds reduce beta to this value. Bigger beta - faster converg
# zeta = 0.015                              # Same for zeta

# beta = 0.1                              # After ~10 seconds reduce beta to this value. Bigger beta - faster converg
zeta = 0.0                              # Same for zeta


def MadgwickQuaternionUpdate6DOF(ax, ay, az, gx, gy, gz, q, deltat, gbias):
    #     float
    # q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3]; // short
    # name
    # local
    # variable
    # for readability
    # float norm; // vector norm
    # float f1, f2, f3; // objetive funcyion elements
    # float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian elements
    # float qDot1, qDot2, qDot3, qDot4;
    # float hatDot1, hatDot2, hatDot3, hatDot4;
    # float gerrx, gerry, gerrz, gbiasx, gbiasy, gbiasz; // gyro bias error

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    # // Auxiliary variables to avoid repeated arithmetic
    _halfq1 = 0.5 * q1
    _halfq2 = 0.5 * q2
    _halfq3 = 0.5 * q3
    _halfq4 = 0.5 * q4
    _2q1 = 2.0 * q1
    _2q2 = 2.0 * q2
    _2q3 = 2.0 * q3
    _2q4 = 2.0 * q4
    _2q1q3 = 2.0 * q1 * q3
    _2q3q4 = 2.0 * q3 * q4

    # // Normalise accelerometer measurement
    norm = sqrt(ax * ax + ay * ay + az * az)
    # // handle NaN
    if norm == 0.0:
        return
    norm = 1.0 / norm
    ax *= norm
    ay *= norm
    az *= norm

    # // Compute the objective function and Jacobian
    f1 = _2q2 * q4 - _2q1 * q3 - ax
    f2 = _2q1 * q2 + _2q3 * q4 - ay
    f3 = 1.0 - _2q2 * q2 - _2q3 * q3 - az
    J_11or24 = _2q3
    J_12or23 = _2q4
    J_13or22 = _2q1
    J_14or21 = _2q2
    J_32 = 2.0 * J_14or21
    J_33 = 2.0 * J_11or24

    # // Compute the gradient(matrix multiplication)
    hatDot1 = J_14or21 * f2 - J_11or24 * f1
    hatDot2 = J_12or23 * f1 + J_13or22 * f2 - J_32 * f3
    hatDot3 = J_12or23 * f2 - J_33 * f3 - J_13or22 * f1
    hatDot4 = J_14or21 * f1 + J_11or24 * f2

    # // Normalize the gradient
    norm = sqrt(hatDot1 * hatDot1 + hatDot2 * hatDot2 + hatDot3 * hatDot3 + hatDot4 * hatDot4)
    hatDot1 /= norm
    hatDot2 /= norm
    hatDot3 /= norm
    hatDot4 /= norm

    # // Compute estimated gyroscope biases
    gerrx = _2q1 * hatDot2 - _2q2 * hatDot1 - _2q3 * hatDot4 + _2q4 * hatDot3
    gerry = _2q1 * hatDot3 + _2q2 * hatDot4 - _2q3 * hatDot1 - _2q4 * hatDot2
    gerrz = _2q1 * hatDot4 - _2q2 * hatDot3 + _2q3 * hatDot2 - _2q4 * hatDot1

    # // Compute and remove gyroscope biases
    gbias[0] += gerrx * deltat * zeta
    gbias[1] += gerry * deltat * zeta
    gbias[2] += gerrz * deltat * zeta
    gx -= gbias[0]
    gy -= gbias[1]
    gz -= gbias[2]

    # // Compute the quaternion derivative
    qDot1 = -_halfq2 * gx - _halfq3 * gy - _halfq4 * gz
    qDot2 = _halfq1 * gx + _halfq3 * gz - _halfq4 * gy
    qDot3 = _halfq1 * gy - _halfq2 * gz + _halfq4 * gx
    qDot4 = _halfq1 * gz + _halfq2 * gy - _halfq3 * gx

    # // Compute then integrate estimated quaternion derivative
    q1 += (qDot1 - (beta * hatDot1)) * deltat
    q2 += (qDot2 - (beta * hatDot2)) * deltat
    q3 += (qDot3 - (beta * hatDot3)) * deltat
    q4 += (qDot4 - (beta * hatDot4)) * deltat

    # // Normalize the quaternion
    norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)  # // normalise quaternion
    norm = 1.0 / norm
    q[0] = q1 * norm
    q[1] = q2 * norm
    q[2] = q3 * norm
    q[3] = q4 * norm
    return q, gbias

TWO_PI = 2 * math.pi
def mod(a, b):
    return a % b


def get_angles_from_quaternion(q):
    # q = [q[1], q[2], q[3], q[0]]
    # atan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy2 - 2 * qz2)
    # yaw = atan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1]**2 - 2 * q[2]**2)
    # pitch = asin(2 * q[0] * q[1] + 2 * q[2] * q[3])
    # roll = atan2(2*q[0]*q[3]-2*q[1]*q[2], 1 - 2*q[0]**2 - 2*q[2]**2)

    yaw = atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)
    pitch = -asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
    #
    # yaw = atan2(2.0 * (q[1] * q[2] - q[0] * q[3]), 2.0 * (q[0]**2 + q[1]**2) - 1)
    # pitch = -asin(2.0 * (q[1] * q[3] + q[0] * q[2]))
    # roll = atan2(2.0 * (q[2] * q[3] - q[0] * q[1]), 2.0 * (q[0]**2 + q[3]**2) - 1)

    # yaw = mod(yaw + TWO_PI, TWO_PI)
    # roll = mod(roll + TWO_PI, TWO_PI)
    # pitch = mod(pitch + TWO_PI, TWO_PI)

    pitch *= 180.0 / math.pi
    yaw *= 180.0 / math.pi
    roll *= 180.0 / math.pi

    # yaw = 360 + yaw if yaw < 0 else yaw

    # yaw = yaw if yaw > 0 else yaw + 360.0
    # roll = roll if roll > 0 else roll + 360.0
    # pitch = pitch if pitch > 0 else pitch + 360.0

    return [pitch, roll, yaw]


def change_quaternion_coordinate_system(q):
    q_new = [0, 0, 0, 0]

    q_new[0] = q[3]
    q_new[1] = -q[0]
    q_new[2] = -q[1]
    q_new[3] = q[2]

    return q_new


def get_angles_from_quaternion2(q):
    # q = [q[1], q[2], q[3], q[0]]
    # atan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy2 - 2 * qz2)
    # yaw = atan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1]**2 - 2 * q[2]**2)
    # pitch = asin(2 * q[0] * q[1] + 2 * q[2] * q[3])
    # roll = atan2(2*q[0]*q[3]-2*q[1]*q[2], 1 - 2*q[0]**2 - 2*q[2]**2)

    # yaw = atan((2.0 * (q[1] * q[2] + q[0] * q[3])) / (q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2))
    # pitch = -asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    # roll = atan((2.0 * (q[0] * q[1] + q[2] * q[3])) / (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2))

    yaw = atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)
    pitch = -asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)

    # yaw = mod(yaw + TWO_PI, TWO_PI)
    # roll = mod(roll + TWO_PI, TWO_PI)
    # pitch = mod(pitch + TWO_PI, TWO_PI)

    pitch *= 180.0 / math.pi
    yaw *= 180.0 / math.pi
    roll *= 180.0 / math.pi

    # yaw = yaw if yaw > 0 else yaw + 360.0
    yaw = 360 - yaw if yaw < 0 else yaw
    # roll = roll if roll > 0 else roll + 360.0
    # pitch = pitch if pitch > 0 else pitch + 360.0

    return [pitch, roll, yaw]


def MadgwickQuaternionUpdate9DOF(ax, ay, az, gx, gy, gz, mx, my, mz, q, deltat):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    _2q1 = 2.0 * q1
    _2q2 = 2.0 * q2
    _2q3 = 2.0 * q3
    _2q4 = 2.0 * q4
    _2q1q3 = 2.0 * q1 * q3
    _2q3q4 = 2.0 * q3 * q4
    q1q1 = q1 * q1
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q1q4 = q1 * q4
    q2q2 = q2 * q2
    q2q3 = q2 * q3
    q2q4 = q2 * q4
    q3q3 = q3 * q3
    q3q4 = q3 * q4
    q4q4 = q4 * q4

    # // Normalise accelerometer measurement
    norm = sqrt(ax * ax + ay * ay + az * az);
    if norm == 0.0:
        return

    norm = 1.0 / norm
    ax *= norm
    ay *= norm
    az *= norm

    # // Normalise magnetometer measurement
    norm = sqrt(mx * mx + my * my + mz * mz)
    if norm == 0.0:
        return
    norm = 1.0 / norm
    mx *= norm
    my *= norm
    mz *= norm

    # // Reference direction of Earth's magnetic field
    _2q1mx = 2.0 * q1 * mx
    _2q1my = 2.0 * q1 * my
    _2q1mz = 2.0 * q1 * mz
    _2q2mx = 2.0 * q2 * mx
    hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
    hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
    _2bx = sqrt(hx * hx + hy * hy)
    _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
    _4bx = 2.0 * _2bx
    _4bz = 2.0 * _2bz

    # // Gradient decent algorithm corrective step
    s1 = -_2q3 * (2.0 * q2q4 - _2q1q3 - ax) + _2q2 * (2.0 * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
    s2 = _2q4 * (2.0 * q2q4 - _2q1q3 - ax) + _2q1 * (2.0 * q1q2 + _2q3q4 - ay) - 4.0 * q2 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az) + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
    s3 = -_2q1 * (2.0 * q2q4 - _2q1q3 - ax) + _2q4 * (2.0 * q1q2 + _2q3q4 - ay) - 4.0 * q3 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
    s4 = _2q2 * (2.0 * q2q4 - _2q1q3 - ax) + _2q3 * (2.0 * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
    norm = sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    # // normalise step magnitude
    norm = 1.0 / norm
    s1 *= norm
    s2 *= norm
    s3 *= norm
    s4 *= norm

    # // Compute rate of change of quaternion
    qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - beta * s1
    qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - beta * s2
    qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - beta * s3
    qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - beta * s4

    # // Integrate to yield quaternion
    q1 += qDot1 * deltat
    q2 += qDot2 * deltat
    q3 += qDot3 * deltat
    q4 += qDot4 * deltat
    norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)    #// normalise quaternion
    norm = 1.0 / norm
    q[0] = q1 * norm
    q[1] = q2 * norm
    q[2] = q3 * norm
    q[3] = q4 * norm

    return q


def PerformMadgwickQuaternion9DOF(a, g, m, time):
    q = [1.0, 0.0, 0.0, 0.0]
    angles = []

    for i in range(len(a[0])):
        dT = time[i+1] - time[i]
        q = MadgwickQuaternionUpdate9DOF(a[0][i], a[1][i], a[2][i],
                                                     g[0][i], g[1][i], g[2][i],
                                                     m[0][i], m[1][i], m[2][i],
                                                     q, dT)
        angles.append(get_angles_from_quaternion(q))

    return [i for i in zip(*angles)]


def PerformMadgwickQuaternion6DOF(a, g, time):
    q = [1.0, 0.0, 0.0, 0.0]
    gbias = [0.0, 0.0, 0.0]
    angles = []

    for i in range(len(a[0])):
        dT = time[i+1] - time[i]
        q, gbias = MadgwickQuaternionUpdate6DOF(a[0][i], a[1][i], a[2][i],
                                                     g[0][i], g[1][i], g[2][i],
                                                     q, dT, gbias)
        # q = change_quaternion_coordinate_system(q)
        angles.append(get_angles_from_quaternion(q))

    return [i for i in zip(*angles)]


def PerformMadgwickQuaternion6DOFOriginal(a, g, time):
    q = [1.0, 0.0, 0.0, 0.0]
    gbias = [0.0, 0.0, 0.0]
    angles = []

    for i in range(len(a[0])):
        dT = time[i+1] - time[i]

        q = MadgwickAHRSupdateIMU(a[0][i], a[1][i], a[2][i],
                                  g[0][i], g[1][i], g[2][i],
                                  q=q, deltat=dT, beta=0.041)

        # q, gbias = MadgwickQuaternionUpdate6DOF(a[0][i], a[1][i], a[2][i],
        #                                              g[0][i], g[1][i], g[2][i],
        #                                              q, dT, gbias)
        # q = change_quaternion_coordinate_system(q)
        angles.append(get_angles_from_quaternion(q))

    return [i for i in zip(*angles)]


def PerformMahonyQuaternion6DOF(a, g, time):
    q = [1.0, 0.0, 0.0, 0.0]
    integralFB = [0.0, 0.0, 0.0]
    angles = []

    for i in range(len(a[0])):
        dT = time[i+1] - time[i]

        q, integralFB = MahonyAHRSupdateIMU(a[0][i], a[1][i], a[2][i],
                                  g[0][i], g[1][i], g[2][i],
                                  q=q, deltat=dT, integralFB=integralFB)
        angles.append(get_angles_from_quaternion(q))

    return [i for i in zip(*angles)]

def to_continuous(data):
    continuous = 0
    ret = []
    prev = 0
    for curr in data:

        if curr < -90 and prev > 90:
            # TODO calculate actual difference if worth... cause will be around 0 anyway ±0.1
            diff = 0
        elif curr > 90 and prev < -90:
            diff = 0
        else:
            diff = curr - prev

        prev = curr
        continuous += diff
        ret.append(continuous)
    return ret


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


# best to be used to find single peaks and not repeatable ones
def get_lag_info(y, lag):
    medFilter = [0]*len(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)

    medFilter[lag - 1] = np.median(y[:lag])
    avgFilter[lag - 1] = np.mean(y[:lag])
    stdFilter[lag - 1] = np.std(y[:lag])

    for i in range(lag, len(y)):
        medFilter[i] = np.median(y[(i - lag):i])
        avgFilter[i] = np.mean(y[(i - lag):i])
        stdFilter[i] = np.std(y[(i - lag):i])

    return dict(medFilter = np.asarray(medFilter),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))
