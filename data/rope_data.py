"""Interface to the baxter data (todo)"""

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import database
import itertools
import window
import numpy as np
import scipy
import random
import math
from cropping_utils import RandomCropper

poke_data_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
DATASET_LOCATION = poke_data_dir + 'datasets/'

RESIZE_SIZE = 240
# X_MIN = 150
# Y_MIN = 110
# SQUARE_SIZE = 300
X_MIN = 150
Y_MIN = 80
SQUARE_SIZE = 400
rc = RandomCropper(X_MIN, Y_MIN, SQUARE_SIZE, RESIZE_SIZE)

img_mean = np.load(open(poke_data_dir + "img_mean.npy"))
cropped_mean = rc.resize(img_mean)

# create a filter to filter out the non-table parts
table_filter = np.zeros((240, 240))
for y in range(240):
    for x in range(240):
        if y > 25 and y > 2.5 * x - 450:
            table_filter[y, x] = 1

SEGMENTATION_SIZE = 20
def resize_segmentation(img):
    return scipy.misc.imresize(img, (SEGMENTATION_SIZE, SEGMENTATION_SIZE))

def rectify(img):
    """Transpose database image to [Y, X, channels] (480x640x3)"""
    return np.transpose(img, [1, 2, 0])

def get_runs(runs, stop=True):
    while True:
        random.shuffle(runs)
        for r in runs:
            db_images = database.ImageLMDB(r + "/image")
            db_pokes = database.SensorLMDB(r + "/poke")
            print "loading from", r, "size", db_pokes.i,
            yield db_images.images(), db_pokes.readings()
        if stop:
            break

def get_rope_segmentation(normalized_image):
    im = np.sum(abs(normalized_image), 2) > 100
    return np.logical_and(im, table_filter)

def get_data(dataset, shuffle=True):
    """dataset: name
    """
    location = DATASET_LOCATION + dataset
    db_image_before = database.ImageLMDB(location + "/image_before", convert_from_ros = False)
    db_image_after = database.ImageLMDB(location + "/image_after", convert_from_ros = False)
    db_poke = database.SensorLMDB(location + "/poke")

    while True:
        iterators = [db_image_before.images(), db_image_after.images(), db_poke.readings()]
        for image_before, image_after, poke in itertools.izip(*iterators):
            n_before = normalize(image_before)
            n_after = normalize(image_after)
            before_segment = get_rope_segmentation(n_before)
            after_segment = get_rope_segmentation(n_after)

            imgs = [n_before, n_after, before_segment, after_segment]
            c_data = rc.random_crop(imgs, poke[:4, 0, 0])
            if c_data:
                cropped_images, cropped_poke = c_data
                img_before, img_after, s_before, s_after = cropped_images
                s_b = resize_segmentation(s_before)
                s_a = resize_segmentation(s_after)
                im = np.sum(abs(img_before.astype(float) - img_after.astype(float)), 2) > 100
                diff = np.sum(im) # rough measure of number of pixels different between the two images
                if diff > 1000:
                    yield img_before, img_after, s_b, s_a, cropped_poke

# up until Sep 8
# TRAIN_DATA = [DATA + "run_rope_" + str(i) for i in range(3, 10)] # 9832
# TEST_DATA = [DATA + "run_rope_" + str(i) for i in range(10, 11)] # 579

TRAIN_DATA = "rope9/train"
TEST_DATA = "rope9/test"

if os.path.exists(DATASET_LOCATION): # don't fail import on machine where data not present
    print('Data loaded')
    print(TRAIN_DATA)
    SOURCES = {"train": get_data(TRAIN_DATA), "val": get_data(TEST_DATA)}

def load_dataset(name):
    print "Loading dataset:", name
    SOURCES["train"] = get_data(name + "/train")
    # SOURCES["val"] = get_data(name + "/test")

def get_size(dataset):
    s = 0
    for run in dataset:
        db1 = database.ImageLMDB(run + "/image")
        db2 = database.ImageLMDB(run + "/poke")
        print run, db1.i, db2.i
        s += db1.i
    return s

def normalize(image):
    # return (image.astype(float) - 127.0) / 100.0
    return image.astype(float) - cropped_mean

distrib = np.zeros((11))
def get_batch(source_name, batch_size = 1):
    """X is the input to the network, Y is the supervision/ground truth"""
    X1 = np.zeros((batch_size, 200, 200, 3))
    X2 = np.zeros((batch_size, 200, 200, 3))
    Y1 = np.zeros((batch_size, 400))
    Y2 = np.zeros((batch_size, 36))
    Y3 = np.zeros((batch_size, 10))
    Z  = np.ones((batch_size)) # ignore length flag
    S1 = np.zeros((batch_size, SEGMENTATION_SIZE, SEGMENTATION_SIZE))
    S2 = np.zeros((batch_size, SEGMENTATION_SIZE, SEGMENTATION_SIZE))
    # Y = np.zeros((batch_size, max_len))
    for example, i in itertools.izip(SOURCES[source_name], xrange(batch_size)):
        image_before, image_after, s_before, s_after, poke = example
        X1[i, :, :, :] = image_before
        X2[i, :, :, :] = image_after
        S1[i, :, :] = s_before
        S2[i, :, :] = s_after
        x, y, theta, length = poke
        # if length > 0.1005 and length < 0.1006:
        if length > 0.1:
            Z[i] = 0
            distrib[10] += 1
        Y1[i, :] = window.encode_pixel(x, y)[0]
        Y2[i, :] = window.encode_theta(theta)[0]
        l_arr, l_ind = window.encode_length(length)
        Y3[i, :] = l_arr
        distrib[l_ind] += 1
    # print distrib
    return X1, X2, Y1, Y2, Y3, S1, S2, Z

if __name__ == "__main__":
    # for example, i in itertools.izip(SOURCES["train"], xrange(30000)):
    #     image_before, image_after, length, pixel, theta = example
    #     print pixel, window.encode_pixel(*pixel)[1]
    #     if i % 1000 == 0:
    #         print i
    t = 0
    # for image_before, image_after, poke in get_data(TRAIN_DATA):
    #     t += 1
    #     print poke
    # print t

    # for i, _ in enumerate(get_data(TEST_DATA, True)):
    #     pass
    s = get_size(TRAIN_DATA)
    print "train data size", s
    # for i, _ in enumerate(get_data(TRAIN_DATA, True)):
    #     pass
    s = get_size(TEST_DATA)
    print "test data size", s
