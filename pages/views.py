from django.shortcuts import render

# Create your views here.
from pathlib import Path
from time import time

from django.core.files.storage import FileSystemStorage
from PIL import Image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import tensorflow as tf

from .models import *


base_path = Path(__file__).resolve().parent.parent
# path_model = base_path / 'models/model_8_3.h5'
path_model = base_path / 'models/efficientnet_512.hdf5'
model = get_efficientnet_unet((512, 512, 3))
# model.summary()
model.load_weights(path_model)


def home_view(request):
	return render(request, 'home.html')


def upload_view(request):
	path_upload = base_path / 'media'
	path_upload.mkdir(parents=True, exist_ok=True)
	path_processed = path_upload / 'processed_images'
	path_processed.mkdir(parents=True, exist_ok=True)

	@nb.jit(nopython=True)
	def conv_2_net(img_norm, v, r, c, d):
		image_batch = np.zeros((r * c, v, v, d), dtype=np.float32)
		index = 0

		for i in range(r):
			for j in range(c):
				image_batch[index, :, :, :] = img_norm[i * v : (i + 1) * v, j * v : (j + 1) * v, :]
				index += 1

		return image_batch


	@nb.jit(nopython=True)
	def post_proc(image, mask, v, r, c):
		line = -1

		for h in range(mask.shape[0]):
			if h % r == 0: # [0, 8]
				line += 1

			for i in range(v):
				for j in range(v):
					if mask[h, i, j, 0] <= 0.5:
						image[line * v + i, h % c * v + j, :] = 0

		return image


	def process_image(input_path, output_path):
		image = np.array(Image.open(input_path))
		# image_resized = np.array(Image.open('path/image.png').resize((256, 256)))
		image_f32 = image.astype(dtype=np.float32)
		image_normalized = image_f32 / 255.0

		w, h, d = image.shape
		v = 512
		r = w // v
		c = h // v

		# image_batch = np.zeros((r * c, v, v, d), dtype=np.float32)
		# index = 0
		#
		# for i in range(r):
		# 	for j in range(c):
		# 		image_batch[index, :, :, :] = image_normalized[i * v : (i + 1) * v, j * v : (j + 1) * v, :]
		# 		index += 1
		#
		# mask = model.predict(image_batch)
		# line = -1
		#
		# for h in range(mask.shape[0]):
		# 	if h % r == 0: # [0, 8]
		# 		line += 1
		#
		# 	for i in range(v):
		# 		for j in range(v):
		# 			if mask[h, i, j, 0] <= 0.5:
		# 				image[line * v + i, h % c * v + j, :] = 0

		image_batch = conv_2_net(image_normalized, v, r, c, d)
		mask = model.predict(image_batch)

		# mask = np.copy(image_batch)
		#
		# for idx in range(mask.shape[0]):
		# 	mask[idx:idx + 1, :, :, :] = model.predict(image_batch[idx:idx + 1, :, :, :])

		img_out = post_proc(image, mask, v, r, c)

		out_image = Image.fromarray(np.uint8(img_out[0 : r * v, 0 : c * v, :]), 'RGB')
		out_image.save(output_path)


	if request.method == 'POST' and 'photo' in request.FILES:
		image_file = request.FILES['photo']
		# print(image_file.name, image_file.size)
		image_name = image_file.name
		fss = FileSystemStorage()

		if Path.exists(path_upload / image_file.name):
			Path.unlink(path_upload / image_file.name)
			fss.save(image_file.name, image_file)
		else:
			fss.save(image_file.name, image_file)

		with tf.device('/GPU:0'):
			start_time = time()
			input_path = path_upload / image_name
			image_name = image_name.split('.')[0] + '_processed.jpg'
			output_path = path_processed / image_name
			process_image(input_path, output_path)

	context = {
		'filename': image_file.name,
		'filename_processed': image_name,
		'time': round(time() - start_time, 2),
	}

	return render(request, 'upload.html', context)
