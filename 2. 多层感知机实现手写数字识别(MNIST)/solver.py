""" 定义训练和测试流程，无需了解 """

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

TRAIN_NUM = 55000 # Training
VAL_NUM = 5000 # Validation
TEST_NUM = 10000 # Test


def train(model, criterion, optimizer, dataset, max_epoch, batch_size, disp_freq):
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []
	get_next = dataset.batch(TRAIN_NUM).make_one_shot_iterator().get_next()
	config = tf.ConfigProto(device_count={'GPU': 0})
	with tf.Session(config=config) as sess:
		# split raw training data(60000) to train_set(55000) and val_set(5000)
		tmp1, tmp2 = sess.run(get_next)
		tmp1 = tf.data.Dataset.from_tensor_slices(tmp1)
		tmp2 = tf.data.Dataset.from_tensor_slices(tmp2)
		train_data = tf.data.Dataset.zip((tmp1, tmp2)).repeat(max_epoch).batch(batch_size)
		# prepare training batch loader
		train_get_next = train_data.make_one_shot_iterator().get_next()

		tmp3, tmp4 = sess.run(get_next)
		tmp3 = tf.data.Dataset.from_tensor_slices(tmp3)
		tmp4 = tf.data.Dataset.from_tensor_slices(tmp4)
		valid_data = tf.data.Dataset.zip((tmp3, tmp4)).repeat(max_epoch).batch(batch_size)
		# prepare validate batch loader
		valid_get_next = valid_data.make_one_shot_iterator().get_next()

		# Training process
		for epoch in range(max_epoch):
			batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_get_next,
			                                                    max_epoch, batch_size, disp_freq, epoch, sess)
			batch_val_loss, batch_val_acc = validate(model, criterion, valid_get_next, batch_size, sess)

			avg_train_acc.append(np.mean(batch_train_acc))
			avg_train_loss.append(np.mean(batch_train_loss))
			avg_val_acc.append(np.mean(batch_val_acc))
			avg_val_loss.append(np.mean(batch_val_loss))

			print()
			print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
				epoch, avg_train_loss[-1], avg_train_acc[-1]))

			print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
				epoch, avg_val_loss[-1], avg_val_acc[-1]))
			print()

	return model, avg_val_loss, avg_val_acc



def train_one_epoch(model, criterion, optimizer, data_get_next, max_epoch, batch_size, disp_freq, epoch, sess):
	batch_train_loss, batch_train_acc = [], []

	max_train_iteration = TRAIN_NUM // batch_size

	for iteration in range(max_train_iteration):
		# Get training data and label
		train_x, train_y = sess.run(data_get_next)

		# Forward pass
		logit = model.forward(train_x)
		criterion.forward(logit, train_y)

		# Backward pass
		delta = criterion.backward()
		model.backward(delta)

		# Update weights, see optimize.py
		optimizer.step(model)

		# Record loss and accuracy
		batch_train_loss.append(criterion.loss)
		batch_train_acc.append(criterion.accu)

		if iteration % disp_freq == 0:
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
				epoch, max_epoch, iteration, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc)))
	return batch_train_loss, batch_train_acc


def validate(model, criterion, data_get_next, batch_size, sess):
	batch_val_acc, batch_val_loss = [], []
	max_val_iteration = VAL_NUM // batch_size

	for iteration in range(max_val_iteration):
		# Get validating data and label
		val_x, val_y = sess.run(data_get_next)

		# Only forward pass
		logit = model.forward(val_x)
		loss = criterion.forward(logit, val_y)

		# Record loss and accuracy
		batch_val_loss.append(criterion.loss)
		batch_val_acc.append(criterion.accu)

	return batch_val_loss, batch_val_acc


def test(model, criterion, dataset, batch_size, disp_freq):
	print('Testing...')
	max_test_iteration = TEST_NUM // batch_size

	batch_test_acc = []
	test_iter = dataset.batch(batch_size).make_one_shot_iterator()
	get_next = test_iter.get_next()
	config = tf.ConfigProto(device_count={'GPU': 0})

	with tf.Session(config=config) as sess:
		for iteration in range(max_test_iteration):
			test_x, test_y = sess.run(get_next)

			# Only forward pass
			logit = model.forward(test_x)
			loss = criterion.forward(logit, test_y)

			batch_test_acc.append(criterion.accu)

	print("The test accuracy is {:.4f}.\n".format(np.mean(batch_test_acc)))
	return np.mean(batch_test_acc)
