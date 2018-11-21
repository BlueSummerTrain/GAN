#encoding=utf-8

import tensorflow as tf
import numpy as np
import data_helpers
from gan_model import GANModel
import os
import datetime

#Parameters
#================================================

#Data loading params
flags = tf.app.flags
flags.DEFINE_string("true_data_file", "./data/train_pos.txt","Data source for the postive data")

#Model hyperparameters

flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
flags.DEFINE_string("padding_token", '<unk>', "uniform sentences")
flags.DEFINE_integer("max_sentences_length",64, "Input sentences max length (default: 64)")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

FLAGS = flags.FLAGS
print "training params:::"
print flags.FLAGS.__flags

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

# Output directory for models and summaries
out_dir = data_helpers.mkdir_if_not_exist("./runs")

# Load true_sentences and build vocab
true_sentences = data_helpers.read_and_clean_file(FLAGS.true_data_file)
padding_true_sentences = data_helpers.padding_sentences(true_sentences,
                                                        FLAGS.padding_token, FLAGS.max_sentences_length)

# Question: should we build voc just use true sentences or use all chinese word dic?
# Here we use true sentences
voc,voc_size = data_helpers.build_vocabulary(padding_true_sentences,'./runs/vocab')

true_data = np.array(data_helpers.sentence2matrix(true_sentences,FLAGS.max_sentences_length,voc))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(true_sentences)))
true_data_shuffled = true_data[shuffle_indices]


#fake_factors = fake_factor_dist.sample((FLAGS.batch_size, FLAGS.max_sentences_length,FLAGS.embedding_dim))

global_graph = tf.Graph()

with global_graph.as_default():
	sess = tf.Session(graph=global_graph)
	gan_model = GANModel(batch_size=FLAGS.batch_size,
	                     sequences_max_length=FLAGS.max_sentences_length,
	                     vocab_size=voc_size,
	                     embedding_size=FLAGS.embedding_dim)
	d_opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
		.minimize(gan_model.d_loss, var_list=gan_model.d_vars)
	r_opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
		.minimize(gan_model.r_loss, var_list=gan_model.r_vars)

	# Build fake_data's factor , R can use it to build fake data
	# Fake data should be same like true_data_embedding
	# Here has some distributions type we can chose
	fake_factor_dist = tf.contrib.distributions.Normal(0., 1.)

	global_step = tf.Variable(0,name="global_step", trainable=False)
	d_loss_summary = tf.summary.scalar('d_loss',gan_model.d_loss)
	r_loss_summaty = tf.summary.scalar('r_loss', gan_model.r_loss)
	train_summary_d_op = tf.summary.merge([d_loss_summary])
	train_summary_r_op = tf.summary.merge([r_loss_summaty])
	train_summary_dir = os.path.join(out_dir, "summaries", "train")
	train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
	checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model"))
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
	# Initialize all variables
	sess.run(tf.global_variables_initializer())


	def train_detector_step(x,z):
		"""
		:param x: real data
		:param z: fake data
		:return: d_loss
		"""
		print type(x)
		print type(z)
		feed_dict = {gan_model.input_x:x,
		             gan_model.input_z:z}
		_,step,summaries,loss = sess.run([d_opt,global_step,train_summary_d_op,gan_model.d_loss],
		                                 feed_dict=feed_dict)
		time_str = datetime.datetime.now().isoformat()
		print("train set:*** {}: step {}, d_loss {:g}".format(time_str, step, loss))
		train_summary_writer.add_summary(summaries, step)
		return loss

	def train_reconstructor_step(z):
		"""
		:param z: fake data
		:return: r_loss
		"""
		feed_dict = {gan_model.input_z:z}
		_,step,summaries,loss = sess.run([r_opt,global_step,train_summary_r_op,gan_model.r_loss],
		                                 feed_dict=feed_dict)
		time_str = datetime.datetime.now().isoformat()
		print("train set:*** {}: step {}, r_loss {:g}".format(time_str, step, loss))
		train_summary_writer.add_summary(summaries, step)
		return loss

	# Generate batches
	x_batches = data_helpers.batch_iter(
		list(zip(true_data_shuffled)), FLAGS.batch_size, FLAGS.num_epochs)
	for batch in x_batches:
		current_step = tf.train.global_step(sess, global_step)
		x_batch = zip(*batch)
		x_batch = np.reshape(np.array(zip(*batch)),[-1,FLAGS.max_sentences_length])
		z_batch = fake_factor_dist.sample((FLAGS.batch_size, FLAGS.max_sentences_length,FLAGS.embedding_dim,1))
		z_batch = z_batch.eval(session=sess)
		print type(z_batch)
		print z_batch.shape
		for j in range(5):
			train_detector_step(x_batch,z_batch)
		train_reconstructor_step(z_batch)
		if current_step % 500 == 0:
			tf.train.write_graph(sess.graph_def, checkpoint_dir, 'gan_cls.pbtxt')
			saver.save(sess, checkpoint_dir + '/gan_cls.ckpt', global_step=current_step)
			print "save ckpt file ok current_step ::: %d" % current_step

















