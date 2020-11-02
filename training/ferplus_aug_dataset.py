#!/usr/bin/python3
import os
import sys

from tqdm import tqdm
from cv2 import cv2
import numpy as np
from corruptions import motion_blur
from corruptions import zoom_blur, pixelate, defocus_blur, gaussian_noise, gaussian_blur, saturate, contrast_plus, contrast, brightness_plus, brightness_minus, elastic_transform, spatter, jpeg_compression, shot_noise
from PIL import Image


class MyCustomAugmentation():
    def __init__(self, corruption_types, corruption_qtys):
        self.corruption_types = corruption_types
        self.corruption_qtys = corruption_qtys
        assert(len(corruption_types)==len(corruption_qtys))
    def __str__(self):
        if max(self.corruption_qtys) != min(self.corruption_qtys):
            s=[]
            for t,q in zip(self.corruption_types, self.corruption_qtys):
                s.append( "%s.%d" % (t.__name__, q) )
            return '.'.join(s)
        else:
            return '.'.join([t.__name__ for t in self.corruption_types]) + "." + str(self.corruption_qtys[0])
    def before_cut(self, img, roi):
        for t,q in zip(self.corruption_types, self.corruption_qtys):
            #print(t,q)
            if q > 0:
                img = t(img, q)
            if len(img.shape)<3:
                img = np.expand_dims(img,2)
            if img.dtype != np.uint8:
                img = img.clip(0,255).astype(np.uint8)
        #print(img.shape, img.dtype)
        return img
    def augment_roi(self, roi):
        return roi
    def after_cut(self, img):
        return img



def contrast_brightness_plus(x, severity):
    sb, sc = [(1,1), (2,1), (2,2), (2,3), (3,4)][severity-1]
    return contrast(brightness_plus(x, sb), sc)
def contrast_brightness_minus(x, severity):
    sb, sc = [(1,1), (2,1), (2,2), (2,3), (3,4)][severity-1]
    return contrast(brightness_minus(x, sb), sc)
    
def gaussian_noise_contrast_brightness_minus(x, severity):
    sg, sb, sc = [(1,1,1), (2,2,1), (2,2,2), (3,2,3), (3,2,4)][severity-1]
    return contrast(brightness_minus(gaussian_noise(x, sg), sb), sc)

def pixelate_contrast_brightness_minus(x, severity):
    sp, sb, sc = [(1,1,1), (2,2,1), (3,2,2), (4,2,1), (4,3,3)][severity-1]
    return contrast(brightness_minus(pixelate(x, sp), sb), sc)
    
def motion_blur_contrast_brightness_minus(x, severity):
    sm, sb, sc = [(2,1,1), (3,1,1), (4,2,2), (5,2,1), (5,2,3)][severity-1]
    return contrast(brightness_minus(motion_blur(x, sm), sb), sc)

corruptions=[
    [contrast_plus],
    [contrast,],
    [brightness_plus,],
    [brightness_minus,],
    [contrast_brightness_plus],
    [contrast_brightness_minus],
    [gaussian_blur],
    [zoom_blur,],
    [defocus_blur,],
    [motion_blur,],
    [gaussian_noise,],
    [shot_noise],
    [spatter,],
    [pixelate,],
    #[contrast, brightness, gaussian_noise,],
    #[contrast, brightness, gaussian_noise,motion_blur],
    #[contrast, brightness, gaussian_noise,spatter],
    #[contrast, brightness, gaussian_noise,motion_blur,spatter],
    [jpeg_compression],
    [gaussian_noise_contrast_brightness_minus],
    [pixelate_contrast_brightness_minus],
    [motion_blur_contrast_brightness_minus]
]

