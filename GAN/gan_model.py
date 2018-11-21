#encoding=utf-8

import tensorflow as tf
import numpy as np
from detector import DecModel
from reconstructor import RecModel
import tensorflow.contrib.layers as ly


class GANModel(object):

	def __init__(self,batch_size,sequences_max_length, vocab_size, embedding_size):
		"""
		:param batch_size: size of one batch 
		:param sequences_max_length: the input height
		:param vocab_size: sentences' word dic size
		:param embedding_size: one word size ,the input width
		"""
		#self.sess = sess
		self.batch_size = batch_size
		self.sequences_max_length = sequences_max_length
		self.input_x = tf.placeholder(tf.int32, [None, sequences_max_length], name='input_true_data')
		self.embedding_size = embedding_size
		#self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input_labels')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')


		#embedding layer,make sentences index to embedding
		#example: sentence dict representation is [1,2,3], the output will be [[*],[*],[*]]
		#output shape will be [sequences_max_length,embedding_size]
		with tf.name_scope("embedding"):
			self.W = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -0.1, 0.1), name="W")
			self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1,name='embedded_chars_expanded')
		print "embedded_chars_expanded.shape"
		print self.embedded_chars_expanded.shape
		self.input_z = tf.placeholder(tf.float32, self.embedded_chars_expanded.shape,name = 'input_fake_factor')

		#build R and D structrue
		self.R = self.reconstructor(self.input_z)
		self.D, self.D_logits = self.detector(self.embedded_chars_expanded, reuse=False)
		self.D_, self.D_logits_ = self.detector(self.R, reuse=True)

		#define loss functions
		def sigmoid_cross_entropy_with_logits(x, y):
			try:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
			except:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

		self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
		self.r_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
		self.d_loss = self.d_loss_real + self.d_loss_fake

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.r_vars = [var for var in t_vars if 'r_' in var.name]

		self.saver = tf.train.Saver()

	def reconstructor(self,z,y=None):
		with tf.variable_scope("reconstructor") as scope:

			"""Encoder::: 3*CNN layers"""
			z_encode = ly.conv2d(z, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm , scope= 'r_en1')
			z_encode = ly.conv2d(z_encode, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm, scope = 'r_en2')
			z_encode = ly.conv2d(z_encode, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm, scope= 'r_en3')

			"""Decoder:::3 * CNN_transpose layers"""
			z_decode = ly.conv2d_transpose(z_encode, 8, 3, stride=2, activation_fn=tf.nn.relu,
				                               normalizer_fn=ly.batch_norm, padding="SAME",
				                               weights_initializer=tf.random_normal_initializer(0, 0.1),scope ='r_de1')
			z_decode = ly.conv2d_transpose(z_decode, 8, 3, stride=2, activation_fn=tf.nn.relu,
				                            normalizer_fn=ly.batch_norm, padding="SAME",
				                            weights_initializer=tf.random_normal_initializer(0, 0.1),scope ='r_de2')
			z_decode = ly.conv2d_transpose(z_decode, 1, 3, stride=2, activation_fn=tf.nn.relu,
				                            normalizer_fn=ly.batch_norm, padding="SAME",
				                            weights_initializer=tf.random_normal_initializer(0, 0.1),scope='r_de3')
			return z_decode

	def detector(self, input_data, reuse=False):
		with tf.variable_scope("detector") as scope:
			"Detector has the same structure as R's encoder"
			if reuse:
				scope.reuse_variables()
			d_h1 = ly.conv2d(input_data, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm , scope= 'd_h1')
			d_h2 = ly.conv2d(d_h1, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm , scope= 'd_h2')
			d_h3 = ly.conv2d(d_h2, 8, kernel_size=3, activation_fn=tf.nn.relu,
				                     normalizer_fn=ly.batch_norm , scope= 'd_h3')
			print tf.reshape(d_h3, [-1, self.sequences_max_length*self.embedding_size*8]).shape
			logit = ly.fully_connected(tf.reshape(d_h3, [-1, self.sequences_max_length*self.embedding_size*8]),
			                           1,activation_fn=None, scope= 'd_logit')

			return tf.nn.sigmoid(logit), logit
