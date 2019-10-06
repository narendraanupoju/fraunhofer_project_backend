# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:42:14 2019

@author: narendra
"""

import cv2

def process(path):
    data = path
    mean= data.mean()
    max_mean = mean*100
    data[data>max_mean]= max_mean
    return data