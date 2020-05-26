import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
#import tensorflow.keras.layers as kl
#(s1, s2, num_features, num_actions, num_neurons, num_hidden_layers):

class DQN(Model):
	"""
	DQN value and target networks
	"""
	def __init__(self, n_inputs, n_outputs, trainable, board = 0,
		n_hlayers = 2, n_neurons = 64, softmax = False, name='FF'):

		super(DQN, self).__init__(name=name)
		self.trainable = trainable
		#self._initialize_variables(n_inputs, n_outputs, n_hlayers, n_neurons,trainable)
		#self.trainable_variablesD = [self.w, self.b]
		self.n_hlayers = n_hlayers
		self.softmax = softmax
		self.net_layers =[]
		Bias_init= tf.keras.initializers.Constant(value=0.1)
		self.net_layers.append(tf.keras.layers.Dense(units=n_inputs, bias_initializer=Bias_init,activation='relu'))
		for i in range (1, n_hlayers+1):
			self.net_layers.append(tf.keras.layers.Dense(units=n_neurons, activation='relu'))
		if self.softmax:
			self.last_layer = tf.keras.layers.Dense(units = n_outputs, activation ="softmax")
		else:
			self.last_layer = tf.keras.layers.Dense(units = n_outputs, activation ="relu")

		self.FLAG = True

	def build(self, input_shape):
	 	super(DQN, self).build(input_shape)

	def call(self, s):
		s = tf.dtypes.cast(s, tf.float32)
		# print("shape of input", s.shape)
		output = s
		for layer in self.net_layers:
			output = layer(output)
		if self.FLAG:
			self.FLAG = False
			self.trainable_variables_d = self.last_layer.trainable_variables
			for layer in self.net_layers:
				self.trainable_variables_d+= layer.trainable_variables
		return self.last_layer(output)


	def get_weights(self):
			return_value = []
			for layer in self.net_layers + [self.last_layer]:
				return_value.append(layer.get_weights())
			return return_value

	def set_weights(self, weights):
			layer_list = self.net_layers + [self.last_layer]
			for i, parameter in enumerate(weights):
				layer_list[i].set_weights(parameter)


class MLPs(Model):
	"""
	DQN value and target networks
	"""
	def __init__(self, n_inputs, n_outputs, trainable, board = 0,
		n_hlayers = 2, n_neurons = 64, softmax = False, name='FF'):

		super(MLPs, self).__init__(name=name)
		self.trainable = trainable
		self._initialize_variables(n_inputs, n_outputs, n_hlayers, n_neurons,
									trainable)
		self.trainable_variablesD = [self.w, self.b]
		self.n_hlayers = n_hlayers
		self.softmax = softmax

	def _initialize_variables(self, n_inputs, n_outputs, n_hlayers, n_neurons,
								trainable):
		rand_trun = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
		# Training Net
		self.w = {0: tf.Variable(initial_value = rand_trun(
													[n_inputs, n_neurons],
													dtype=tf.float32),
										trainable = trainable)}
		self.b = {0: tf.Variable(initial_value = tf.constant(0.1,
													shape = [n_neurons],
													dtype = tf.float32),
										trainable = trainable)}
		for i in range(1, n_hlayers+1):
			self.w[i] = tf.Variable(initial_value = rand_trun(
														[n_neurons, n_neurons],
														dtype = tf.float32),
											trainable = trainable)
			self.b[i] = tf.Variable(initial_value = tf.constant(0.1,
														shape=[n_neurons],
														dtype=tf.float32),
											trainable = trainable)
		self.w[n_hlayers+1] = tf.Variable(initial_value = rand_trun(
													[n_neurons, 128],
													dtype=tf.float32),
												trainable = trainable)
		self.b[n_hlayers+1] = tf.Variable(initial_value = tf.constant(0.1,
															shape = [128],
															dtype = tf.float32),
												trainable = trainable)
		self.w[n_hlayers+2] = tf.Variable(initial_value = rand_trun(
													[128, n_outputs],
													dtype=tf.float32),
												trainable = trainable)
		self.b[n_hlayers+2] = tf.Variable(initial_value = tf.constant(0.1,
															shape = [n_outputs],
															dtype = tf.float32),
												trainable = trainable)
	def forward(self, s):
		s = s.astype(np.float32)
		# print("shape of input", s.shape)
		layer = tf.matmul(s, self.w[0]) + self.b[0]
		layer = tf.nn.relu(layer)
		for i in range(1, self.n_hlayers+2):
			layer = tf.matmul(layer, self.w[i]) + self.b[i]
			layer = tf.nn.relu(layer)
		values = tf.matmul(layer, self.w[self.n_hlayers+2]) +\
			self.b[self.n_hlayers+2]
		if self.softmax: values = tf.nn.softmax(values)

		return values
