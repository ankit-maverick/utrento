import numpy as np, h5py
import cPickle as pickle
import sPickle

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# transpose, shuffle and pickle
print "Training data..."
f = h5py.File('train_X.mat','r')
train_X = f.get('data_matrix')
train_X = np.array(train_X)
train_X = np.round(train_X.T).astype(np.int16)
train_X.shape

f = h5py.File('train_Y.mat','r')
train_Y = f.get('label_vector')
train_Y = np.array(train_Y)
train_Y = np.squeeze(train_Y)
train_Y.shape

train_X, train_Y = shuffle_in_unison_inplace(train_X, train_Y)
valid_X = train_X[-50000:]
valid_Y = train_Y[-50000:]
train_X = train_X[:-50000]
train_Y = train_Y[:-50000]


print "Test data..."

f = h5py.File('test_X.mat','r')
test_X = f.get('data_matrix')
test_X = np.array(test_X)
test_X = test_X.T
test_X.shape

f = h5py.File('test_Y.mat','r')
test_Y = f.get('label_vector')
test_Y = np.array(test_Y)
test_Y = np.squeeze(test_Y)
test_Y.shape


print "Test bordi data..."

f = h5py.File('test_bordi_X.mat','r')
test_bordi_X = f.get('data_matrix')
test_bordi_X = np.array(test_bordi_X)
test_bordi_X = test_bordi_X.T
test_bordi_X.shape

f = h5py.File('test_bordi_Y.mat','r')
test_bordi_Y = f.get('label_vector')
test_bordi_Y = np.array(test_bordi_Y)
test_bordi_Y = np.squeeze(test_bordi_Y)
test_bordi_Y.shape

print "Test omogenee data..."

f = h5py.File('test_omogenee_X.mat','r')
test_omogenee_X = f.get('data_matrix')
test_omogenee_X = np.array(test_omogenee_X)
test_omogenee_X = test_omogenee_X.T
test_omogenee_X.shape

f = h5py.File('test_omogenee_Y.mat','r')
test_omogenee_Y = f.get('label_vector')
test_omogenee_Y = np.array(test_omogenee_Y)
test_omogenee_Y = np.squeeze(test_omogenee_Y)
test_omogenee_Y.shape



datasets = ((train_X, train_Y), (valid_X, valid_Y),
			(test_X, test_Y), (test_bordi_X, test_bordi_Y),
			(test_omogenee_X, test_omogenee_Y))

print "Starting to pickle..."
pickle.dump(datasets, open("datasets_p.pkl", "wb" ) )

