from __future__ import division
import os
#import magic
import urllib.request
from app import app
from flask_cors import CORS, cross_origin
from flask import Flask, flash, request, redirect, render_template, jsonify, json, make_response
from werkzeug.utils import secure_filename
import io
import numpy as np
from werkzeug.local import Local, LocalProxy
from werkzeug.datastructures import ImmutableMultiDict
import base64
from PIL import Image, ImageFile
import numpy as np
from Database import insertBLOB, convertToBinaryData
import sqlite3
import cv2
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import matplotlib.pyplot as plt

sys.setrecursionlimit(40000)
#img_path = test_path

def normalize(x):
	min_val = np.min(x)
	max_val = np.max(x)
	x = (x- min_val)/(max_val - min_val)
	return x

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)



@app.route('/uploadimage', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
	print("request is ", request.files)
	st = time.time()
	content_length = request.content_length
	print(f"Content_length : {content_length}")
	print("data type is ", type(request))
	print("data type of request files  ", type(request.files))
	data_dict = request.form.to_dict()
	#print(type(data_dict))
	#print(data_dict['file'])
	#print('data from frontend',data_dict)
	data = (data_dict['file'].split(',')[1])
	l, b = (data_dict['imgDimensions'].split(','))
	l = int(l)
	b = int(b)
	print('width of image', l)
	print('type of l ',type(l))
	print('height of image', b)
	#print(data)
	#print(len(data_dict))
	#print(data)
	imgdata = base64.b64decode(data)
	print("imagedata type is" , type(imgdata))
	img2 = Image.open(io.BytesIO(imgdata))
	print(type(img2))
	#img2.show()
	#img = cv2.imread(img2)
	#print('opencv type' , type(img))
	#print(type(img))
	a = np.array(img2.getdata()).astype(np.float64)
	#print('datatype of w ', w.dtype)
	#b = np.ones(172800,3)
	#a = np.concatenate((w,b), axis=None)
	print('type of data to model ', type(a))
	print('shape of data from frontend', a.shape)
	#r, c = a.shape
	#print('Value of r', r)
	"""
	if a.shape == (480000, 3):
		data = a.reshape(600, 800, 3)
	else: data = a.reshape(480, 640, 3)
	"""
	data = a.reshape(b,l,3)

	st = time.time()

	parser = OptionParser()

	parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=64)
	parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
	parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

	(options, args) = parser.parse_args()

	config_output_filename = options.config_filename

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	if C.network == 'resnet50':
		import keras_frcnn.resnet as nn
	elif C.network == 'vgg':
		import keras_frcnn.vgg as nn

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.items()}
	print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = int(options.num_rois)

	if C.network == 'resnet50':
		num_features = 1024
	elif C.network == 'vgg':
		num_features = 512

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, num_features)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)

	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)

	model_classifier = Model([feature_map_input, roi_input], classifier)

	print('Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	all_imgs = []

	classes = {}

	bbox_threshold = 0.6

	visualise = True

	#if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):	
	#	continue
	#print(img_name)
	#filepath = os.path.join(img_path,img_name)

	img = data

	#cv2.imshow('img', img)
	#cv2.waitKey(0)

	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.6)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.6)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	print('Elapsed time = {}'.format(time.time() - st))
	print('number of windoiws detected',len(all_dets))
	print(all_dets)
	r = len(all_dets)
	img3 = normalize(img)
	#plt.imshow(img)
	#cv2.imshow('img3', img3)
	#cv2.waitKey(0)
	
	K.clear_session()
	#data = process(data)
	#print('after reshape',data.shape)
	im2 = Image.fromarray(img.astype("uint8"),"RGB")
	print("im2 data type is " , type(im2))
	#to_frontend = (" ".join(str(x) for x in data))
	db = data.tobytes()
	print('type of data to database :', type(db))
	todb = insertBLOB('Image007', db)
	print('final data shape fed to model : ', data.shape)
	# ImageFile img = db.b64encode()
	# with open("t.png", "rb") as imageFile:
    # str = base64.b64encode(imageFile.read())
	#cv2.imshow('image', cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))
	#cv2.waitKey()
	#str = base64.b64encode(data)
	# return jsonify(to_frontend, r)

	#img = Image.open( 'C:\Window Counter_Project\Flickr\Window_101 (131).jpg' )
	#img.load()

	#data = np.asarray( img, dtype="int32" )
	#im = Image.fromarray(data.astype("uint8"))
	#im.show()  # uncomment to look at the image
	rawBytes = io.BytesIO()
	print(rawBytes)
	im2.save(rawBytes, "jpeg")
	#im2.show()
	print('type of im2 is ',type(im2))
	rawBytes.seek(0)  # return to the start of the file
	response_obj = {
	'count': r,
	'image':"data:image/jpeg;base64,"+str(base64.b64encode(rawBytes.read()))
	}
	#print("response is", type(response_obj))
	return jsonify(Data=response_obj)


if __name__ == "__main__":
    app.run(debug=True)
