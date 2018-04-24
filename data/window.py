import numpy as np
import random
import math
import bisect

IN = 240
OUT = 240

def crop(img, x, y):
    new_img = img[:, y:y+OUT, x:x+OUT]
    return new_img

def shift_and_normalize(action, dx, dy):
    action[0, 0, 0] *= IN
    action[0, 0, 0] -= dx
    action[1, 0, 0] *= IN
    action[1, 0, 0] -= dy

    x, y = action
    ret = encode_pixel(x, y)

    action[0, 0, 0] /= OUT
    action[1, 0, 0] /= OUT
    return ret, action, ind

XY_BINS = 20
dx = 1.0 / XY_BINS
def encode_pixel(x, y):
    """Expect x, y from 0-1"""
    # TODO: this should really be an assert
    x = min(0.9999, max(0, x))
    y = min(0.9999, max(0, y))

    arr = np.zeros((XY_BINS * XY_BINS))
    x, y = int(x/dx), int(y/dx)
    x, y = min(max(0, x), XY_BINS-1), min(max(0, y), XY_BINS-1)
    ind = XY_BINS * x + y
    arr[ind] = 1
    return arr, ind

# LENGTH_BINS = 10
# dl = 0.04 / LENGTH_BINS
# def encode_length(l):
#     """Length in meters"""
#     l = min(0.04999, max(0.01, l))

#     arr = np.zeros((LENGTH_BINS))
#     ind = int((l - 0.01)/dl)
#     arr[ind] = 1
#     return arr, ind

# LENGTH_BINS = 10
# MIN_L = 0.01
# MAX_L = 0.149999
# dl = (MAX_L - MIN_L) / LENGTH_BINS
# def encode_length(l):
#     """Length in meters"""
#     l = min(MAX_L, max(MIN_L, l))

#     arr = np.zeros((LENGTH_BINS))
#     ind = int((l - MIN_L)/dl)
#     arr[ind] = 1
#     return arr, ind

LENGTH_BINS = 10
# LENGTH_BINS_SORTED = [0.031354602426290512, 0.05083392933011055, 0.059834115207195282, 0.074937090277671814, 0.090499997138977051, 0.1005, 0.1006, 0.11, 0.11050000041723251, 0.15] # obtained by binning the first 10000 pokes uniformly
LENGTH_BINS_SORTED = [0.020790038630366325, 0.03206624835729599, 0.04270794615149498, 0.051202502101659775, 0.056311529129743576, 0.062374196946620941, 0.072047881782054901, 0.081927493214607239, 0.091911002993583679, 0.15]
def encode_length(l):
    """Length in meters binned by a fixed array"""
    arr = np.zeros((LENGTH_BINS))
    ind = bisect.bisect_right(LENGTH_BINS_SORTED, l)
    if ind == 10:
        # print "UH OH length", l
        ind = 9

    arr[ind] = 1
    return arr, ind

THETA_BINS = 36
dt = 2 * math.pi / THETA_BINS
def encode_theta(t):
    arr = np.zeros((THETA_BINS))
    ind = int(t / dt)
    arr[ind] = 1
    return arr, ind

# class WindowLayer(caffe.Layer):

#     def setup(self, bottom, top):
#         # check input pair
#         pass

#     def reshape(self, bottom, top):
#         N = bottom[0].num # batch size

#         top[0].reshape(2 * N, 3, OUT, OUT)
#         top[1].reshape(N, XY_BINS * XY_BINS, 1, 1)
#         top[2].reshape(N, 2, 1, 1)
#         top[3].reshape(N, 2, 1, 1)
#         top[4].reshape(N, LENGTH_BINS)
#         top[5].reshape(N, THETA_BINS)
#         top[6].reshape(N, 1)
#         top[7].reshape(N, 1)
#         top[8].reshape(N, 1)

#     def forward(self, bottom, top):
#         N = bottom[0].num # batch size
#         for i in range(N):
#             img_before = bottom[0].data[i, :, :, :]
#             img_after = bottom[1].data[i, :, :, :]
#             action = np.zeros((2, 1, 1))
#             action[:2, :, :] = np.copy(bottom[2].data[i, :2, :, :])
#             x, y = bottom[2].data[i, :, 0, 0]
#             length = bottom[3].data[i, 0, 0, 0]
#             theta = bottom[4].data[i, 0, 0, 0]

#             # dx = random.randint(max(1, int(x) - 200), max(1, min(250, int(x-30))))
#             # dy = random.randint(max(1, int(y) - 200), max(1, min(250, int(y-30))))

#             dx = random.randint(0, 40)
#             dy = random.randint(0, 40)

#             top[0].data[i, :, :, :] = crop(img_before, dx, dy)
#             top[0].data[N + i, :, :, :] = crop(img_after, dx, dy)
#             array, action, xy_ind = shift_and_normalize(action, dx, dy)
#             top[1].data[i, :, 0, 0] = array

#             jitter = np.zeros((2, 1, 1))
#             jitter[0, 0, 0] = dx
#             jitter[1, 0, 0] = dy
#             top[2].data[i, :, :, :] = jitter

#             top[3].data[i, :, :, :] = action

#             top[4].data[i, :], length_ind = encode_length(length)

#             top[5].data[i, :], theta_ind = encode_theta(theta)

#             top[6].data[i, 0] = xy_ind
#             top[7].data[i, 0] = length_ind
#             top[8].data[i, 0] = theta_ind

#     def backward(self, top, propagate_down, bottom):
#         pass
