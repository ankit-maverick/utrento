import numpy as np
import time
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv



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
		assert image_shape[1] = filter_shape[1]
		self.input = input

		# Size of the receptive field for a hidden neuron
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])

		fan_out = 
		W_values = np.asarray(rng.uniform())

