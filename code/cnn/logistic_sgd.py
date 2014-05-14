import theano
import time
import numpy as np
import theano.tensor as T


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


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	index = T.lscalar() # index to a minibatch
	x = T.matrix('x')
	y = T.ivector('y')

	classifier = LogisticRegression(x, 28*28, 10)
	cost = classifier.negative_log_likelihood(y)

	test_model = theano.function(inputs=[index], outputs=classifier.errors(y), given={x: test_set_x})
