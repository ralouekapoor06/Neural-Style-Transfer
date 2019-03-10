import os
import sys
import scipy.io 
#to read and write data to a variety of file formats
import scipy.misc
#other scipy classes
import matplotlib.pyplot as plt 
from PIL import image
import cv2
from nst_utils import *
import numpy as np 
import tensorflow as tf 

model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")
#model is a python dictionary containing variable and its value
