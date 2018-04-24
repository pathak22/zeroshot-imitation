"""Cropping utils"""

import itertools
import window
import numpy as np
import cv2
import random

CROP_SIZE = 200
def crop(img, x, y):
    """Crop img with offsets x and y"""
    if len(img.shape) == 3:
        new_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE, :]
    if len(img.shape) == 2:
        new_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
    return new_img

class RandomCropper:
    def __init__(self, ox, oy, square, r, dx=None, dy=None):
        """Params of random cropping:
        ox = offset x
        oy = offset y
        square = initial square crop size
        r = resize square size
        dx, dy = constant offset
        Then each image is further randomly cropped to CROP_SIZE.
        """
        self.params = [ox, oy, square, r]
        self.dx = dx
        self.dy = dy

    def random_crop(self, images, poke):
        interior = False
        for _ in range(10):
            dx = self.dx or random.randint(0, 40)
            dy = self.dy or random.randint(0, 40)
            poke_cropped = self.resized_cropped_poke(poke, dx, dy)
            if self.is_interior_poke(poke_cropped):
                cropped_images = []
                for image in images:
                    cropped_images.append(crop(image, dx, dy))
                poke_cropped = self.resized_cropped_poke(poke, dx, dy)
                return cropped_images, poke_cropped
        return None

    def resize(self, img):
        """Resizes img into r x r by first cropping a square x square offset by (ox, oy)"""
        ox, oy, square, r = self.params
        return cv2.resize(img[oy:oy+square, ox:ox+square, :], (r, r))

    def resized_cropped_poke(self, poke, dx, dy):
        ox, oy, square, r = self.params
        x, y, theta, length = poke
        x_new = ((x - ox) / float(square) * r - dx) / float(CROP_SIZE)
        y_new = ((y - oy) / float(square) * r - dy) / float(CROP_SIZE)
        return x_new, y_new, theta, length


    def cropped_poke_to_real(self, poke, dx, dy):
        ox, oy, square, r = self.params
        x, y, theta, length = poke
        x_new = ((x * CROP_SIZE) + dx) * square / float(r) + ox
        y_new = ((y * CROP_SIZE) + dy) * square / float(r) + oy
        return x_new, y_new, theta, length

    def is_interior_poke(self, poke):
        x, y, _, _ = poke
        return all([x < 0.9, x > 0.1, y < 0.9, y > 0.1])
