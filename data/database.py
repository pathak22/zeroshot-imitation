import cv2
#from cv_bridge import CvBridge, CvBridgeError
import lmdb
import caffe
import numpy as np
import scipy
import itertools

#bridge = CvBridge()

def save_vector(vector, label, db):
    x = vector
    if len(vector.shape) != 3:
        x = vector[:, np.newaxis, np.newaxis]
    datum = caffe.io.array_to_datum(x, long(label))
    str_id = '{:08}'.format(label)
    with db.begin(write=True) as txn:
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

def save_img(data, label, db, size = [227, 227, 3], transpose = True, scale = 1.0, convert_from_ros = False):
    if convert_from_ros:
        cv_image = bridge.imgmsg_to_cv2(data).astype(int)
    else:
        cv_image = data
    x = np.asarray(cv_image) * scale
    x = x.astype(np.uint8)
    if len(x.shape) == 2:
        x = x[:, :, None]
    if size:
        # n = x.shape[2]
        # s = size[:].append(n)
        x = scipy.misc.imresize(x, s)
    if transpose:
        x = x.transpose()
    datum = caffe.io.array_to_datum(x, 0)
    str_id = '{:08}'.format(label)
    with db.begin(write=True) as txn:
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

class ImageLMDB(object):
    def __init__(self, name, size = None, scale = 1, convert_from_ros = True):
        self.name = name
        self.DB = lmdb.open(name, 99999999999)
        self.i = self.get_num_elements()
        self.size = size
        self.scale = scale
        self.convert_from_ros = convert_from_ros

    def save(self, data, label = None):
        if not label:
            label = self.i
        save_img(data, label, self.DB, self.size, scale = self.scale, convert_from_ros = self.convert_from_ros)
        self.i += 1

    def get_num_elements(self):
        return int(self.DB.stat()['entries'])

    def play(self, millis = 20, transpose = True, start = 0, end = 0, key = 0):
        cv2.namedWindow(self.name)
        img_size = None
        with self.DB.begin() as txn:
            cursor = txn.cursor()
            # print "got here"
            # label = '{:08}'.format(start).encode('ascii')
            # cursor.set_key(label)
            if start:
                for _ in xrange(start):
                    cursor.next()
            elif end:
                cursor.last()
                for _ in xrange(end):
                    cursor.prev()
            elif key:
                cursor = txn.cursor()
                label = '{:08}'.format(key).encode('ascii')
                print "key", cursor.set_key(label)
            for key, img_datum in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(img_datum)
                img = caffe.io.datum_to_array(datum) # .astype(np.uint8)
                if not img_size:
                    print img.shape
                    img_size = img.shape
                if transpose:
                    img = img.transpose()
                cv2.imshow(self.name, img)
                cv2.waitKey(millis)
        return img_size

    def info(self):
        print self.DB.stat()

    def images(self, start = 0, transpose = True):
        with self.DB.begin() as txn:
            cursor = txn.cursor()
            label = '{:08}'.format(start).encode('ascii')
            cursor.set_key(label)
            for key, img_datum in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(img_datum)
                img = caffe.io.datum_to_array(datum).astype(np.uint8)
                if transpose:
                    img = img.transpose()
                yield img

    def iterator(self):
        return self.images()

    def rollback_once(self):
        self.i = self.i - 1
        with self.DB.begin(write=True) as txn:
            label = '{:08}'.format(self.i).encode('ascii')
            txn.delete(label)

class SensorLMDB(object):
    def __init__(self, name):
        self.name = name
        self.DB = lmdb.open(name, 99999999999)
        self.i = self.get_num_elements()

    def save(self, data, label = None):
        if not label:
            label = self.i
        save_vector(data, label, self.DB)
        self.i += 1

    def get_num_elements(self):
        return int(self.DB.stat()['entries'])

    def play(self):
        with self.DB.begin() as txn:
            cursor = txn.cursor()
            for key, sensor_datum in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(sensor_datum)
                vec = caffe.io.datum_to_array(datum)
                print vec

    def readings(self, start = 0):
        with self.DB.begin() as txn:
            cursor = txn.cursor()
            label = '{:08}'.format(start).encode('ascii')
            cursor.set_key(label)
            for key, sensor_datum in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(sensor_datum)
                vec = caffe.io.datum_to_array(datum)
                yield vec

    def iterator(self):
        return self.readings()

    def info(self):
        print self.DB.stat()
        print self.DB.info()

    def rollback_once(self):
        self.i = self.i - 1
        with self.DB.begin(write=True) as txn:
            label = '{:08}'.format(self.i).encode('ascii')
            txn.delete(label)

def test():
    db = ImageLMDB("/home/ashvin/data/static/arm_motion", transpose=False)
    db.play()

if __name__ == "__main__":
    # db = ImageLMDB("/home/ashvin/data/poke/depth_after")
    db = ImageLMDB("rope9/train/image_after")
    db2 = SensorLMDB('rope9/train/poke')
    db.info()
    db2.info()
    # db.play(40)

    # vel_db = SensorLMDB("/home/ashvin/data/poke/")
    # pos_db = SensorLMDB("/home/ashvin/data/poke/positions")
    # vel_db.play()
    # pos_db.play()
