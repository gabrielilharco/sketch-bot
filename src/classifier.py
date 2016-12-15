from data_processing import *
from visual_vocabulary import *
from scipy import ndimage
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,cuda.root=/usr/local/cuda,device=gpu0,floatX=float32"
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.model_selection import train_test_split

def build_mlp(input_var=None):
	""" 
	Creates an MLP of two hidden layers of 800 units each, followed by
	a softmax output layer of 250 units. It applies 20% dropout to the input
	data and 50% dropout to the hidden layers
	"""
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
									 input_var=input_var)

	# Apply 20% dropout to the input data:
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Add a fully-connected layer of 800 units, using the linear rectifier, and
	# initializing weights with Glorot's scheme (which is the default anyway):
	l_hid1 = lasagne.layers.DenseLayer(
			l_in_drop, num_units=800,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())

	# We'll now add dropout of 50%:
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

	# Another 800-unit layer:
	l_hid2 = lasagne.layers.DenseLayer(
			l_hid1_drop, num_units=800,
			nonlinearity=lasagne.nonlinearities.rectify)

	# 50% dropout again:
	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

	# Finally, we'll add the fully-connected output layer, of 250 softmax units:
	l_out = lasagne.layers.DenseLayer(
			l_hid2_drop, num_units=250,
			nonlinearity=lasagne.nonlinearities.softmax)

	# Each layer is linked to its incoming layer(s), so we only need to pass
	# the output layer to give access to a network in Lasagne:
	return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
	num_epochs = 500
	# Load dataset
	X = np.load('images.npy')
	X = X.reshape(-1,1,28,28)
	y = np.load('labels.npy')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	network = build_mlp(input_var)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	# Create update expressions for training.
	# We'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()

	# Create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))