import numpy as np
import tensorflow as tf
import random, time, os.path, shutil
from utils.network import DQN, MLPs
from utils.schedules import LinearSchedule
from utils.dfa import *
from utils.game import *
from utils.utils import clear_screen
from tensorflow.python.framework import ops
# from common.schedules import LinearSchedule
import fastrand,math
class I_DQN_l:
	"""
	feedfodward DQN
	"""
	def __init__(self, num_actions, num_features, ltl,
		training_params, obs_proxy, board, policy_name='Main', num_agents = 2):
		# initialize attributes
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.num_features = num_features
		self.training_params = training_params
		self.gamma = self.training_params.gamma
		self.ltl_scope_name = "DQN_" + str(ltl).replace("&","AND").\
			replace("|","OR").replace("!","NOT").replace("(","P1_").\
			replace(")","_P2").replace("'","").replace(" ","").replace(",","_")

		self.policy_name = policy_name # could be of interest later
		#(self, n_inputs, n_outputs, trainable, board = 0,
		#n_hlayers = 2, n_neurons = 64, softmax = False, name='FF'):
		n_neurons = 64
		n_hlayers = 2
		#self.DQN = DQN(self.num_features, self.num_actions, n_neurons, n_hidden_layers)
		#self.DQN_target = DQN(self.num_features, self.num_actions, n_neurons, n_hidden_layers)
		self.DQN = MLPs(num_features, num_actions, trainable = True)
		self.DQN_target =  MLPs(num_features, num_actions, trainable = True)
		#self.DQN.build((None,self.num_features))
		#self.DQN_target.build((None,self.num_features))
		#self.DQN.summary()
		#self.DQN_target.summary()

		# Creating the experience replay buffer
		self.batch_size = training_params.batch_size
		self.learning_starts = training_params.learning_starts
		self.replay_buffer = IDQNReplayBuffer(training_params.replay_size)
		# This proxy adds the machine state representation to the MDP state
		self.obs_proxy = obs_proxy
		self.lr = self.training_params.final_lr
		self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

		# count of the number of environmental steps
		self.step = 0




	def save_transition(self, s1, a, reward, s2, done):
		self.replay_buffer.add(s1, a.value, reward, s2, float(done))

	def learn(self):
		"""
		Minimize the error in Bellman's equation on a batch sampled
		from replay buffer.
		"""
		s1, a, r, s2, done = self.replay_buffer.sample(self.batch_size)
		#print("line 65, learnig sample: ", s2.shape,"\n",s2)
		#input()
		# target, out of gradients
		q_target = self.DQN_target.forward(s2)
		q_max = tf.reduce_max(q_target, axis=-1)
		# print("target.shape", q_target.shape)
		# print("q_max:", q_max)
		q_max = q_max * (1.0-done) # dead ends must have q_max equal to zero
		# agsgs+=1
		q_target_value = r + self.gamma * q_max



		with tf.GradientTape() as tape:
			q_values = self.DQN.forward(s1)

			# Q_values -> get optimal actions
			self.best_action = tf.argmax(q_values, 1)
			# Optimizing with respect to q_target
			action_mask = tf.one_hot(indices=a, depth=self.num_actions,
				dtype=tf.float32)
			q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)

			q_target_value = tf.stop_gradient(q_target_value)

			# Computing td-error and loss function
			loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))
		gradients = tape.gradient(loss, self.DQN.trainable_variables)
		self.optimizer.apply_gradients(\
							zip(gradients, self.DQN.trainable_variables))


	def get_best_action(self, s1):
		q_values = self.DQN.forward(s1)
		best_action = tf.argmax(q_values, -1)
		self.best_action = best_action

		return self.best_action
		#return self.sess.run(self.best_action, {self.s1: s1})

	def update_target_network(self):
		#for dqn, target_dqn in zip(self.DQN, self.DQN_target):
		#print("-----Update target networks weight:")
		#print("DQN: ",np.array(self.DQN.get_weights()).shape)
		#self.DQN_target.set_weights(self.DQN.get_weights())
		for var, var_tar in zip(self.DQN.trainable_weights,
								self.DQN_target.trainable_weights):
			var_tar.assign(var)
		#w = np.array(self.DQN.trainable_weights)
		#print("---size: ", w.shape, "size 2: ", len(w[0].shape)==2)
		#for i, var in enumerate(self.DQN.trainable_weights):
		#	print("----updating target: ", i)
		#	print(var)
		#	input()
		# ###Below is a tesing for the crossover.
		gen1 = np.array(self.DQN.trainable_weights)
		gen2 = np.array(self.DQN_target.trainable_weights)
		for W1, W2 in zip(gen1, gen2):
		# 	#param1 (size of components, ) if (2, ): has the weights and bias. if (1, ) only weight matrix
			if(len(W1.shape) == 2): #weight w_matrix
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
				for i in range(num_cross_overs):
					receiver_choice = random.random()# Choose which gene to receive the perturbation
					if(receiver_choice < 0.5):
						ind_cr = fastrand.pcg32bounded(W1.shape[0])
						W1[ind_cr, :].assign(W2[ind_cr, :])
					else:
						ind_cr = fastrand.pcg32bounded(W1.shape[0])
						W2[ind_cr, :].assign(W1[ind_cr, :])
			elif(len(W1.shape) == 1): #bias vector
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				num_cross_overs = fastrand.pcg32bounded(num_variables)  # Lower bounded on full swaps
				for i in range(num_cross_overs):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if(receiver_choice < 0.5):
						ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
						W1[ind_cr].assign(W2[ind_cr])
					else:
						ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
						W2[ind_cr].assign(W1[ind_cr])

	def get_steps(self):
		return self.step

	def add_step(self):
		self.step += 1

class IDQNReplayBuffer(object):
	def __init__(self, size):
		self._storage = []
		self._maxsize = size
		self._next_idx = 0

	def __len__(self):
		return len(self._storage)

	def add(self, s1, a, r, s2, done):
		#num_features = s1.shape[0]*s1.shape[1]
		#s1 = s1.reshape((1, num_features))
		#s2 = s2.reshape((1, num_features))
		data = (s1, a, r, s2, done)

		if self._next_idx >= len(self._storage):
			self._storage.append(data)
		else:
			self._storage[self._next_idx] = data
		self._next_idx = (self._next_idx + 1) % self._maxsize

	def _encode_sample(self, idxes):
		S1, A, R, S2, DONE = [], [], [], [], []
		for i in idxes:
			data = self._storage[i]
			s1, a, r, s2, done = data
			S1.append(np.array(s1, copy=False))
			A.append(np.array(a, copy=False))
			R.append(r)
			S2.append(np.array(s2, copy=False))
			DONE.append(done)
		return np.array(S1), np.array(A), np.array(R),\
				np.array(S2), np.array(DONE)

	def sample(self, batch_size):
		idxes = [random.randint(0, len(self._storage) - 1) \
				for _ in range(batch_size)]
		return self._encode_sample(idxes)

#Effective extension of the standard MDP state
class Obs_Proxy:
	def __init__(self, env):
		# NOTE: He said he had to add a representations for 'True' and 'False'
		#		(even if they are not important in practice)
		num_states = len(env.dfa.ltl2state) - 2
		ltl2hotvector = {}
		for f in env.dfa.ltl2state:
			if f not in ['True', 'False']:
				aux = np.zeros((num_states), dtype=np.float64)
				aux[len(ltl2hotvector)] = 1.0
				ltl2hotvector[f] = aux
		ltl2hotvector["False"] = np.zeros((num_states), dtype=np.float64)
		ltl2hotvector["True"] = np.zeros((num_states), dtype=np.float64)
		self.ltl2hotvector = ltl2hotvector

	def get_observation(self, env, agent):
		s = env.get_observation(agent)
		  # adding the DFA state to the observation
		s_extended = np.concatenate((s,
				self.ltl2hotvector[env.get_LTL_goal()]))
		return s_extended

def train_DQNs ( DQNs, spec_params, tester, curriculum, show_print, render):
	# Initializing parameters
	dqns = DQNs[spec_params.ltl_spec]
	training_params = tester.training_params
	testing_params = tester.testing_params

	env = Game(spec_params)
	obs_proxy = Obs_Proxy(env)
	agents = env.agents
	action_set = env.get_actions(agents[0]) # NOTE: only if they all have the same action set
	# All the agents have the same observation
	num_features = len(obs_proxy.get_observation(env, env.agents[0]))
	max_steps = training_params.max_timesteps_per_spec
	Replay_buffers = {}
	for agent in agents:
		Replay_buffers[str(agent)] = IDQNReplayBuffer(
			training_params.replay_size)
	exploration = LinearSchedule(
		schedule_timesteps = int(training_params.exploration_frac \
			* max_steps), initial_p=1.0,
		final_p = training_params.final_exploration)

	training_reward = 0
	last_ep_rew = 0
	episode_count = 0  # episode counter
	rew_batch = np.zeros(100)

	if show_print: print("Executing ", max_steps, " steps...")
	if render: env.show_map()

	#We start iterating with the environment
	for t in range (max_steps):
		actions = []
		for agent, dqn in zip(agents.values(), dqns.values()):
			# Getting the current state and ltl goal
			s1 = obs_proxy.get_observation(env, agent)

			# Choosing an action to perform
			if random.random() < exploration.value(t):
				act = random.choice(action_set) # take random actions
			else:
				act = Actions(dqn.get_best_action(s1.reshape((1,num_features))))
				# print("Act", act)
			actions.append(act)
			dqn.add_step()
		# updating the curriculum
		curriculum.add_step()

		# Executing the action
		reward = env.execute_actions(actions)
		if render and episode_count%30 is 0:
			time.sleep(0.01)
			clear_screen()
			env.show_map()

		training_reward += reward

		for agent, dqn, act in zip(agents.values(), dqns.values(),
									actions):
			# Saving this transition
			s2 = obs_proxy.get_observation(env, agent) # adding the DFA state to the features
			done = env.ltl_game_over or env.env_game_over
			dqn.save_transition(s1, act, reward, s2, done)

			# Learning
			if dqn.get_steps() > training_params.learning_starts and \
				dqn.get_steps() % training_params.values_network_update_freq \
				== 0:
				dqn.learn()

			# Updating the target network
			if dqn.get_steps() > training_params.learning_starts and \
				dqn.get_steps() % training_params.target_network_update_freq\
				== 0: dqn.update_target_network()

		# Printing
		if show_print and (dqns['0'].get_steps()+1) \
							% training_params.print_freq == 0:
			print("Step:", dqns['0'].get_steps()+1, "\tTotal reward:",
				last_ep_rew, "\tSucc rate:",
				"%0.3f"%curriculum.get_succ_rate(),
				"\tNumber of episodes:", episode_count)

		# Testing
		if testing_params.test and (curriculum.get_current_step() \
				% testing_params.test_freq == 0):
					tester.run_test(curriculum.get_current_step(),
						_test_DQN, DQNs)

		# Restarting the environment (Game Over)
		if done:
			# Game over occurs for one of three reasons:
			# 1) DFA reached a terminal state,
			# 2) DFA reached a deadend, or
			# 3) The agent reached an environment deadend (e.g. a PIT)
			# Restarting
			env = Game(spec_params)
			obs_proxy = Obs_Proxy(env)
			agents = env.agents
			rew_batch[episode_count%100]= training_reward
			episode_count+=1
			last_ep_rew = training_reward
			training_reward = 0

			# updating the hit rates
			curriculum.update_succ_rate(t, reward)
			# Uncomment if want to stop learning according to succ. rate
			# if curriculum.stop_spec(t):
			# 	last_ep_rew = 0
			# 	if show_print: print("STOP SPEC!!!")
			# 	break

		# checking the steps time-out
		if curriculum.stop_learning():
			if show_print: print("STOP LEARNING!!!")
			break

	if show_print:
		print("Done! Last reward:", last_ep_rew)

def _test_DQN(spec_params, training_params, testing_params, DQNs):
	# Initializing parameters
	dqns = DQNs[spec_params.ltl_spec]
	env = Game(spec_params)
	obs_proxy =  Obs_Proxy(env)
	agents = env.agents
	# Starting interaction with the environment
	r_total = 0
	for t in range(testing_params.num_steps):
		actions = []
		for agent, dqn in zip(agents.values(), dqns.values()):
			# Getting the current state and ltl goal
			s1 = obs_proxy.get_observation(env, agent)

			# Choosing an action to perform
			act = Actions(dqn.get_best_action(s1.reshape((1,len(s1)))))
			actions.append(act)
		# Executing the actions
		r_total += env.execute_actions(actions) * training_params.gamma**t

		# Restarting the environment (Game Over)
		if env.ltl_game_over or env.env_game_over:
			break

	return r_total

def _initialize_dqns(training_params, tester):
	dqns = {}
	#initializing one DQN per spec
	board = 0 #for storing tensorboard
	for ltl_spec in tester.get_LTL_specs():
		dqns[ltl_spec] = {}
		env = Game(tester.get_spec_params(ltl_spec))
		obs_proxy = Obs_Proxy(env)
		# For now we assume all agents have the same input and output dimensions
		num_features = len(obs_proxy.get_observation(env, env.agents[0]))
		num_actions  = len(env.get_actions(env.agents[0]))

		# we create a different DQN for each spec, and a different DQN per agent
		for agent in range(env.n_agents):
			dqns[ltl_spec][str(agent)] = I_DQN_l( num_actions, num_features,
				ltl_spec, training_params, obs_proxy, board)
		board+=1
	return dqns

def run_experiments(tester, curriculum, saver, num_times, show_print, render):
	time_init = time.time()
	training_params = tester.training_params
	for t in range(num_times):
		# Setting the random seed to 't'
		random.seed(t)
		tf.random.set_seed(t)
		# # To support multithreading
		# tf_config = tf.ConfigProto()
		# tf_config.inter_op_parallelism_threads=2
		# tf_config.intra_op_parallelism_threads=2
		# # Reduce if want to run multiple experiments in parallel
		# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
		# #sess = tf.Session(config=tf_config, graph=None)

		# Reseting default values
		curriculum.restart()

		# Initializing DQNs, currently one per spec per agent
		DQNs = _initialize_dqns( training_params, tester)

		while not curriculum.stop_learning():
			if show_print: print("Current step:", curriculum.get_current_step(),
			"from", curriculum.total_steps)
			spec = curriculum.get_next_spec()
			spec_params = tester.get_spec_params(spec)
			train_DQNs(DQNs, spec_params, tester, curriculum, show_print,
				render)
		ops.reset_default_graph()
		# Backing up the results
		saver.save_results()

	# Showing results
	tester.show_results()
	print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
