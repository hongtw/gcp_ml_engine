from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python import estimator
import numpy as np

SAMPLE_SIZE = 1000

def getFeatures():
    Average_Score = tf.feature_column.numeric_column('Average_Score')
    lat = tf.feature_column.numeric_column('lat')
    lng = tf.feature_column.numeric_column('lng')
    return [Average_Score,lat ,lng]

def build_estimator(model_dir):
	"""Build an estimator."""
	params = tensor_forest.ForestHParams(
		num_classes=10, num_features=51,
		num_trees=100, max_nodes=1000)
	graph_builder_class = tensor_forest.RandomForestGraphs
	graph_builder_class = tensor_forest.TrainingLossForest
	# Use the SKCompat wrapper, which gives us a convenient way to split
	# in-memory data like MNIST into batches.
	return estimator.SKCompat(random_forest.TensorForestEstimator(
		params, graph_builder_class=graph_builder_class,
		model_dir=model_dir))

def getData(fp):
    train_size = 0.9
    data = np.loadtxt(fp)
    np.random.shuffle(data)
    x = data[:SAMPLE_SIZE, :-1].astype(np.int32)
    y = data[:SAMPLE_SIZE, -1].astype(np.int32)
    print(x.shape, y.shape)
    num_features = 51
    num_classes = len(set(y))
    total = len(x)
    train_size = int(total * train_size)
    return x[:train_size, :], y[:train_size], x[train_size:, :], y[train_size:], num_features, num_classes


def transLabel(y_train, y_test):
	le = LabelEncoder()
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)
	return y_train, y_test, le 


x_train, y_train, x_test, y_test, num_features, num_classes = getData('train.dense')

# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train)
# y_test = lb.transform(y_test)
y_train, y_test, le = transLabel(y_train, y_test)


#specify params
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
	# feature_colums= getFeatures(), 
	num_classes=num_classes, 
	num_features=num_features, 
	regression=False, 
	num_trees=10, 
	max_nodes=1000).fill()
print("Params =")
print(vars(params))

clf = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
    params, model_dir="./model/")
clf.fit(x=x_train, y=y_train)

print(clf.score(x_test, y_test))



# #build the graph
# graph_builder_class = tensor_forest.RandomForestGraphs

# est=random_forest.TensorForestEstimator(
#   params, graph_builder_class=graph_builder_class)

# #define input function
# train_input_fn = numpy_io.numpy_input_fn(
#   x=x_train,
#   y=y_train,
#   batch_size=1000,
#   num_epochs=1,
#   shuffle=True)

# est.fit(input_fn=train_input_fn, steps=500)
