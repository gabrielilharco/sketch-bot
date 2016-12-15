from data_processing import *
from visual_vocabulary import *
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,cuda.root=/usr/local/cuda,device=gpu0,floatX=float32"
import time
import numpy as np
import lasagne
from matplotlib import pyplot
from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne import layers

if __name__ == '__main__':
	net1 = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1', layers.Conv2DLayer),
			('pool1', layers.MaxPool2DLayer),
			('conv2', layers.Conv2DLayer),
			('pool2', layers.MaxPool2DLayer),
			('conv3', layers.Conv2DLayer),
			('pool3', layers.MaxPool2DLayer),
			('hidden4', layers.DenseLayer),
			('hidden5', layers.DenseLayer),
			('output', layers.DenseLayer),
		],
		input_shape=(None, 1, 48, 48),
		conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
		conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
		conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
		hidden4_num_units=1000, hidden5_num_units=1000,
		output_num_units=250, output_nonlinearity=lasagne.nonlinearities.softmax,
		update_learning_rate=0.03,
		update_momentum=0.9,
		regression=False,
		max_epochs=50,
		verbose=1,
		train_split=TrainSplit(0.2, stratify=True)
	)
	X = np.load('images.npy')
	X = X.reshape(-1,1,48,48)
	y = np.load('labels.npy')

	net1.fit(X,y)
	import cPickle as pickle
	with open('net.pickle', 'wb') as f:
		pickle.dump(net, f, -1)

	train_loss = np.array([i["train_loss"] for i in net1.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
	pyplot.plot(train_loss, linewidth=3, label="train")
	pyplot.plot(valid_loss, linewidth=3, label="valid")
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.ylim(1e-3, 1e-2)
	pyplot.yscale("log")
	pyplot.show()