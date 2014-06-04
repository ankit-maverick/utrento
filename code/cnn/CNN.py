import sys
import os
import gzip


import numpy as np
import time
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


# TODO : Setting up optimal hyperparameters
# TODO : Convert the dataset in the desired input format
#        See https://groups.google.com/forum/#!searchin/theano-users/data$20format/theano-users/abA0jV05vuM/2XBMHOk0Q90J


class LogisticRegression(object):

	def __init__(self, input, n_in, n_out):

		# Parameters
		self.W = theano.shared(value=np.zeros((n_in, n_out)), dtype=theano.config.floatx, name='W', borrow=True)
		self.b = theano.shared(value=np.zeros((n_out,)), dtype=theano.config.floatx, name='b', borrow=True)

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution."""
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch ; zero one
		loss over the size of the minibatch

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""

		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',
				('y', target.type, 'y_pred', self.y_pred.type))
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.tanh):
		"""
		rng : numpy.random.RandomState
			a random number generator used to initialize weights

		input : theano.tensor.dmatrix
			A symbolic tensor of shape (n_data_points, n_in)

		n_in : int
			No. of neurons in previous layer(input)

		n_out : int
			No. of neurons in this HiddenLayer

		activation : theano.Op or function
			Non linearity function to be applied
		"""
		self.input = input

		if W is None:
			W_bound = np.sqrt(6. / (n_in + n_out))
			W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
								  size=(n_in, n_out)),
								  dtype=theano.config.floatX)

			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (lin_output if activation is None else activation(lin_output))

		# parameters of the model
		self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		"""
		rng : numpy.random.RandomState
			A random no. generator used to initialize weights

		input : theano.tensor.dtensor4
			symbolic image tensor of shape image_shape

		filter_shape : tuple or list of length 4
			(number of filters, num input feature maps, filter height, filter width)

		image_shape : tuple or list of length 4
			(batch size, num input feature maps, image height, image width)

		poolsize : tuple or list of length 2
			the downsampling factor

		"""
		assert image_shape[1] == filter_shape[1]
		self.input = input

		# Size of the receptive field for a hidden neuron
		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])

		# each unit in the previous layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
				   np.prod(poolsize))

		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound,
							   high=W_bound, size=filter_shape),
							   dtype=theano.config.floatX), borrow=True)

		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(input=input, filters=self.W,
							   filter_shape=filter_shape,
							   image_shape=image_shape)

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize,
											ignore_border=True)

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]


def load_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset
	'''

	#############
	# LOAD DATA #
	#############

	data_dir, data_file = os.path.split(dataset)

	try:
		if data_dir == "" and not os.path.isfile(dataset):
			# Check if dataset is in the data directory.
			new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
			if os.path.isfile(new_path) or data_file == 'dataset.pkl.gz':
				dataset = new_path
		print '... loading data'
		# Load the dataset
		f = gzip.open(dataset, 'rb')
		train_set, valid_set, test_set, test_bordi, test_omogenee = cPickle.load(f)
		f.close()

	except IOError:
		print "Dataset not found in the given path...exiting"
		sys.exit()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)
	test_bordi_x, test_bordi_y = shared_dataset(test_bordi)
	test_omogenee_x, test_omogenee_y = shared_dataset(test_omogenee)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
			(test_set_x, test_set_y), (test_bordi_x, test_bordi_y),
			(test_omogenee_x, test_omogenee_y)]
	return rval


def evaluate_CNN(learning_rate=0.1, n_epochs=200, dataset='dataset.pkl.gz',
				 nkerns=[20, 50], batch_size=100):
	"""
	learning_rate : float
		learning rate used (factor for the stochastic gradient)

	n_epochs : int
		maximal number of epochs to run the optimizer

	dataset : string
		path to the pickled dataset used for training and testing

	nkerns : list of ints
		number of kernels on each layer
	"""

	rng = np.random.RandomState(11)
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	test_bordi_x, test_bordi_y = datasets[3]
	test_omogenee_x, test_omogenee_y = datasets[4]


	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_test_bordi_batches = test_bordi_x.get_value(borrow=True).shape[0]
	n_test_omogenee_batches = test_omogenee_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size
	n_test_bordi_batches /= batch_size
	n_test_omogenee_batches /= batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						# [int] labels

	ishape = (5, 15, 15)  # this is the size of the images

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	layer0_input = x.reshape((batch_size, ishape[0], ishape[1], ishape[2]))

	# Construct the first convolutional pooling layer:
	layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
			 image_shape=(batch_size, 5, 15, 15),
			 filter_shape=(nkerns[0], 5, 4, 4), poolsize=(2, 2))

	# Construct the second convolutional pooling layer:
	layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
			 image_shape=(batch_size, nkerns[0], 6, 6),
			 filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))

	# (nkerns[0], nkerns[1] * 2 * 2)
	layer2_input = layer1.output.flatten(2)

	# construct a fully-connected sigmoidal layer
	layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 2 * 2,
						 n_out=200, activation=T.tanh)

	# classify the values of the fully-connected sigmoidal layer
	layer3 = LogisticRegression(input=layer2.output, n_in=200, n_out=10)

	# the cost we minimize during training is the NLL of the model
	cost = layer3.negative_log_likelihood(y)

	# create a function to compute the mistakes that are made by the model
	test_model = theano.function([index], layer3.errors(y),
			 givens={
				x: test_set_x[index * batch_size: (index + 1) * batch_size],
				y: test_set_y[index * batch_size: (index + 1) * batch_size]})

	test_bordi_model = theano.function([index], layer3.errors(y),
			 givens={
				x: test_bordi_x[index * batch_size: (index + 1) * batch_size],
				y: test_bordi_y[index * batch_size: (index + 1) * batch_size]})

	test_omogenee_model = theano.function([index], layer3.errors(y),
			 givens={
				x: test_omogenee_x[index * batch_size: (index + 1) * batch_size],
				y: test_omogenee_y[index * batch_size: (index + 1) * batch_size]})


	validate_model = theano.function([index], layer3.errors(y),
			givens={
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

	# create a list of all model parameters to be fit by gradient descent
	params = layer3.params + layer2.params + layer1.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)

	# train_model is a function that updates the model parameters by
	# SGD Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i],grads[i]) pairs.
	updates = []
	for param_i, grad_i in zip(params, grads):
		updates.append((param_i, param_i - learning_rate * grad_i))

	train_model = theano.function([index], cost, updates=updates,
		  givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]})


	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
					  (epoch, minibatch_index + 1, n_train_batches, \
					   this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test error of best '
						   'model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

					# test it on the test bordi set
					test_losses = [test_bordi_model(i) for i in xrange(n_test_bordi_batches)]
					test_score = numpy.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test bordi error of best '
						   'model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

					# test it on the test omogenee set
					test_losses = [test_omogenee_model(i) for i in xrange(n_test_omogenee_batches)]
					test_score = numpy.mean(test_losses)
					print(('     epoch %i, minibatch %i/%i, test omogenee error of best '
						   'model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))


			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i,'\
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
	evaluate_CNN()



