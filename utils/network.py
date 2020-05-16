import tensorflow as tf

#(s1, s2, num_features, num_actions, num_neurons, num_hidden_layers):

class DQN(Model):
	"""
	DQN value and target networks
	"""
	def __init__(self, n_inputs, n_outputs, trainable, board = 0,
		n_hlayers = 2, n_neurons = 64, softmax = False, name='FF'):

		super(DQN, self).__init__(name=name)
		self.trainable = trainable
		self._initialize_variables(n_inputs, n_outputs, n_hlayers, n_neurons,
									trainable)
		self.trainable_variablesD = [self.w, self.b]
		self.n_hlayers = n_hlayers
		self.softmax = softmax
		self.layers =[]
		Bias_init= tf.keras.initializers.Constant(value=0.1)
		self.layers.append(tf.keras.layers.Dense(units=n_inputs, bias_initializer=Bias_init,activation='relu'))
		for i in range (1, n_hlayers+1):
			self.layers.append(tf.keras.layers.Dense(units=n_neurons, activation='relu'))
		if self.softmax:
			self.last_layer = tf.keras.layers.Dense(units = n_outputs, activation ="softmax")
		else:
			self.last_layer = tf.keras.layers.Dense(units = n_outputs, activation ="relu")

		self.FLAG = True
	#
	# def build(self, input_shape):
	# 	super(DQN, self).build(input_shape)
	#
	def call(self, s):
		s = tf.dtypes.cast(s, tf.float32)
		# print("shape of input", s.shape)
		output = s
		for layer in self.layers:
			output = layer(output)
		if self.FLAG:
			self.FLAG = False
			self.trainable_variables = self.last_layer.trainable_variables
			for layer in self.layers:
				self.trainable_variables+= layer.trainable_variables
		return self.last_layer(output)


	def get_weights(self):
			return_value = []
			for layer in self.layers + [self.last_layer]:
				return_value.append(layer.get_weights())
			return return_value

	def set_weights(self, weights):
			layer_list = self.layers + [self.last_layer]
			for i, parameter in enumerate(weights):
				layer_list[i].set_weights(parameter)



#
# def get_MLP(s1, s2, num_features, num_actions, num_neurons, num_hidden_layers):
#     # Instructions to update the target network
#     update_target = []
#     # First layer
#     layer, layer_t = _add_layer(s1, s2, num_features, num_neurons, True, update_target, 0)
#     # Hidden layers
#     for i in range(num_hidden_layers):
#         layer, layer_t = _add_layer(layer, layer_t, num_neurons, num_neurons, True, update_target, i+1)
#     # Output Layer
#     q_values, q_target = _add_layer(layer, layer_t, num_neurons, num_actions, False, update_target, num_hidden_layers+1)
#     return q_values, q_target, update_target
#
# def _add_layer(s, s_t, num_input, num_output, use_relu, update, id):
#     layer, W, b = _add_dense_layer(s, num_input, num_output, True, id)
#     layer_t, W_t, b_t = _add_dense_layer(s_t, num_input, num_output, False, id)
#     update.extend([tf.assign(W_t,W), tf.assign(b_t,b)])
#     if use_relu:
#         layer = tf.nn.relu(layer)
#         layer_t = tf.nn.relu(layer_t)
#     return layer, layer_t
#
# def _add_dense_layer(s, num_input, num_output, is_trainable, id):
#     #generated values follow a normal distribution with specified mean and standard deviation
#     W = tf.Variable(tf.truncated_normal([num_input, num_output], stddev=0.1, dtype=tf.float64), trainable = is_trainable, name="W"+str(id))
#     b = tf.Variable(tf.constant(0.1, shape=[num_output], dtype=tf.float64), trainable = is_trainable, name="b"+str(id))
#
#     return tf.matmul(s, W) + b, W, b
