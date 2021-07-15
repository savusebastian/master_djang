from django.db import models

# Create your models here.
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf


def get_unet(input_img, n_filters=8, dropout=0.2):
	# Contracting Path
	c1 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(input_img)
	c1 = BatchNormalization()(c1)
	c1 = Activation('relu')(c1)
	p1 = MaxPooling2D((2, 2))(c1)

	c2 = Conv2D(n_filters * 2, 3, kernel_initializer='he_normal', padding='same')(p1)
	c2 = BatchNormalization()(c2)
	c2 = Activation('relu')(c2)
	p2 = MaxPooling2D((2, 2))(c2)

	c3 = Conv2D(n_filters * 4, 3, kernel_initializer='he_normal', padding='same')(p2)
	c3 = BatchNormalization()(c3)
	c3 = Activation('relu')(c3)
	p3 = MaxPooling2D((2, 2))(c3)

	c4 = Conv2D(n_filters * 8, 3, kernel_initializer='he_normal', padding='same')(p3)
	c4 = BatchNormalization()(c4)
	c4 = Activation('relu')(c4)
	p4 = MaxPooling2D((2, 2))(c4)

	c5 = Conv2D(n_filters * 16, 3, kernel_initializer='he_normal', padding='same')(p4)
	c5 = BatchNormalization()(c5)
	c5 = Activation('relu')(c5)

	# Expansive Path
	u6 = Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding='same')(c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u6)
	c6 = BatchNormalization()(c6)
	c6 = Activation('relu')(c6)

	u7 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding='same')(c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u7)
	c7 = BatchNormalization()(c7)
	c7 = Activation('relu')(c7)

	u8 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding='same')(c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u8)
	c8 = BatchNormalization()(c8)
	c8 = Activation('relu')(c8)

	u9 = Conv2DTranspose(n_filters * 1, 3, strides=(2, 2), padding='same')(c8)
	u9 = concatenate([u9, c1])
	c9 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u9)
	c9 = BatchNormalization()(c9)
	c9 = Activation('relu')(c9)

	outputs = Conv2D(1, 1, activation='sigmoid')(c9)
	model = Model(inputs=[input_img], outputs=[outputs])

	return model


def get_efficientnet_unet(input_shape):
	# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
	# https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302
	i_s = Input(input_shape)

	def c_bn_a(input_shape, filters, kernel_size=3, act=True):
		c2d = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(input_shape)
		bn = tf.keras.layers.BatchNormalization()(c2d)
		a = tf.nn.silu(bn) if act else tf.identity(bn)

		return a


	def s_e(input_shape, filters, r=24):
		if filters < r * 2:
			r = filters

		o = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(input_shape)

		# ap2d = tf.keras.layers.AveragePooling2D()(input_shape)
		ap2d = tf.nn.avg_pool2d(input_shape, 2, 1, 'SAME')
		c2d1 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(ap2d)
		a1 = tf.nn.silu(c2d1)
		c2d2 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(a1)
		a2 = tf.keras.activations.sigmoid(c2d2)

		return o * a2


	def mb_conv_n(input_shape, filters, expansion_factor=1, kernel_size=3, p=0):
		# MBConv with an expansion factor of N, plus squeeze-and-excitation
		expanded = expansion_factor * input_shape.shape[3]

		expand_pw = tf.identity(input_shape) if (expansion_factor == 1) else c_bn_a(input_shape, expanded, kernel_size=1)
		depthwise = c_bn_a(expand_pw, expanded, kernel_size=kernel_size)
		se = s_e(depthwise, filters)
		reduce_pw = c_bn_a(se, filters, kernel_size=1, act=False)

		return reduce_pw


	def up_block(previous_block, contracting_block, filters, kernel_size=3):
		ub1 = Conv2D(filters, kernel_size=kernel_size - 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(previous_block))
		ub2 = Concatenate(axis=3)([contracting_block, ub1])
		ub3 = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(ub2)
		ub4 = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(ub3)

		return ub4


	# Contracting Path
	# Block 1
	bl1 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(i_s)

	# Block 2
	bl2 = mb_conv_n(bl1, 16)
	pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl2)

	# Block 3
	bl3 = mb_conv_n(pool2, 24, expansion_factor=6)
	pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl3)

	# Block 4
	bl4 = mb_conv_n(pool3, 40, expansion_factor=6, kernel_size=5)
	pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl4)

	# Block 5
	bl5 = mb_conv_n(pool4, 80, expansion_factor=6)
	pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl5)

	# Block 6
	bl6 = mb_conv_n(pool5, 112, expansion_factor=6, kernel_size=5)
	pool6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl6)

	# Block 7
	bl7 = mb_conv_n(pool6, 192, expansion_factor=6, kernel_size=5)
	pool7 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl7)

	# Block 8
	bl8 = mb_conv_n(pool7, 320, expansion_factor=6)
	pool8 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(bl8)

	# Block 9
	c9 = tf.keras.layers.Conv2D(1280, kernel_size=1)(bl8)
	p9 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c9)
	# p9 = tf.keras.layers.GlobalAveragePooling2D()(p9)
	bl9 = tf.keras.layers.Dense(2)(p9)

	# Expanding path
	# Expanding Block 8
	ebl8 = up_block(bl9, bl8, 320)

	# Expanding Block 7
	ebl7 = up_block(ebl8, bl7, 192)

	# Expanding Block 6
	ebl6 = up_block(ebl7, bl6, 112)

	# Expanding Block 5
	ebl5 = up_block(ebl6, bl5, 80)

	# Expanding Block 4
	ebl4 = up_block(ebl5, bl4, 40)

	# Expanding Block 3
	ebl3 = up_block(ebl4, bl3, 24)

	# Expanding Block 2
	ebl2 = up_block(ebl3, bl2, 16)

	output = Conv2D(1, 1, activation='sigmoid')(ebl2)

	return Model(i_s, output)
