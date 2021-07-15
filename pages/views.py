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
import tensorflow as tf

from .models import *


base_path = Path(__file__).resolve().parent.parent
# path_model = base_path / 'models/model_8_3.h5'
path_model = base_path / 'models/efficientnet_512.hdf5'
model = get_efficientnet_unet((512, 512, 3))
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
# model.summary()
model.load_weights(path_model)


def home_view(request):
	return render(request, 'home.html')


def upload_view(request):
	path_upload = base_path / 'media'
	path_upload.mkdir(parents=True, exist_ok=True)
	path_processed = path_upload / 'processed_images'
	path_processed.mkdir(parents=True, exist_ok=True)

	def process_image(input_path, output_path):
		image = np.array(Image.open(input_path))
		# image_resized = np.array(Image.open('path/image.png').resize((256, 256)))
		image_f32 = image.astype(dtype=np.float32)
		image_normalized = image_f32 / 255.0

		w, h, d = image.shape
		v = 512
		r = w // v
		c = h // v

		image_batch = np.zeros((r * c, v, v, d), dtype=np.float32)
		index = 0

		for i in range(r):
			for j in range(c):
				image_batch[index, :, :, :] = image_normalized[i * v : (i + 1) * v, j * v : (j + 1) * v, :]
				index += 1

		mask = model.predict(image_batch)
		# mask_reshaped = mask.reshape((r * v, c * v))

		line = -1

		for h in range(mask.shape[0]):
			if h % r == 0: # [0, 8]
				line += 1

			for i in range(v):
				for j in range(v):
					if mask[h, i, j, 0] <= 0.5:
						image[line * v + i, h % c * v + j, :] = 0

		out_image = Image.fromarray(np.uint8(image[0 : r * v, 0 : c * v, :]), 'RGB')
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
