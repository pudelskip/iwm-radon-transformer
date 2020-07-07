import cv2
import numpy as np


def normalize(M):
    M_min = np.min(M)
    M_max = np.max(M)
    M -= M_min
    M /= (M_max - M_min)
    return M


def normalize_255(M):
    M =M.astype(np.float64)
    M_min = np.min(M)
    M -= M_min
    M_max = np.max(M)
    M /= M_max
    M*=255
    return M


def fil(x):
    if x == 0:
        return 1.0
    elif x % 2 == 0:
        return 0.0
    else:
        return (-4.0/ (np.pi * np.pi)) / (x * x)


def scan_line2(x0, y0, x1, y1, img, mode, val):
    line_sum = 0
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2 * dy - dx
    y = y0
    for i, x in enumerate(range(x0, x1)):
        if mode == "back":
            img[x][y] += val
        else:
            line_sum += img[x][y]

        if D > 0:
            y = y + yi
            D = D - 2 * dx
        D = D + 2 * dy
    if mode == "back":
        return 0
    else:
        return line_sum / (x1 - x0)


def scan_line1(x0, y0, x1, y1, img, mode, val):
    line_sum = 0
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2 * dx - dy
    x = x0
    for i, y in enumerate(range(y0, y1)):
        if mode == "back":
            img[x][y] += val
        else:
            line_sum += img[x][y]

        if D > 0:
            x = x + xi
            D = D - 2 * dy
        D = D + 2 * dx

    if mode == "back":
        return 0
    else:
        return line_sum / (y1 - y0)


def scan_line(x0, y0, x1, y1, img, mode="", val=0):
    average = 0
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            average = scan_line2(x1, y1, x0, y0, img, mode, val)
        else:
            average = scan_line2(x0, y0, x1, y1, img, mode, val)
    else:
        if y0 > y1:
            average = scan_line1(x1, y1, x0, y0, img, mode, val)
        else:
            average = scan_line1(x0, y0, x1, y1, img, mode, val)
    return average


def get_y():
    result = [0] * 90
    for i in range(0, 80):
        if 0 <= i <= 9:
            result[i] = 0
            continue
        if 10 <= i <= 46:
            result[i] = 1
            continue
        if 47 <= i <= 65:
            result[i] = 0
            continue
        if 66 <= i <= 77:
            result[i] = 1
            continue
        if 78 <= i <= 89:
            result[i] = 0
            continue
    return result


def custom_convolution(first_sig, second_sig):
    signal_size = len(first_sig)
    second_size = len(second_sig)
    nm1 = signal_size + second_size - 1
    convolve_res = [0] * nm1
    for n in range(0, nm1):
        for m in range(0, n + 1):
            if n - m >= second_size:
                continue
            if m >= signal_size:
                break
            fi = first_sig[m]
            se = second_sig[n - m]
            what_add = fi * se
            convolve_res[n] += what_add
    offset = (second_size - 1) // 2
    return convolve_res[offset:nm1 - offset - 1]
