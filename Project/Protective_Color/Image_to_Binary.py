# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os, glob
from sklearn.model_selection import train_test_split

caltech_dir = "/home/hts221/protective/TRAIN_DIR"
categories = ["color","background"]
nb_classes = len(categories)
image_w = 64
image_h = 64
pixels = image_h*image_w*3

X = []
Y = []

for idx, n in enumerate(categories):
	label = [0 for i in range(nb_classes)]
	label[idx] = 1

	image_dir = caltech_dir+"/"+n
	files = glob.glob(image_dir+"/*.jpg")
	for i, f in enumerate(files):
		img = Image.open(f)
		img = img.convert("RGB")
		img = img.resize((image_w, image_h))
		data = np.asarray(img)
		X.append(data)
		Y.append(label)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
	train_test_split(X,Y)
xy = (X_train, X_test, y_train, y_test)
np.save("/home/hts221/protective/5obj.npy",xy)
print "ok,", len(Y)