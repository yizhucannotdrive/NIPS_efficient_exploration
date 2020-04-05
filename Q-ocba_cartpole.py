import gym
import keras
import numpy as np
import random

from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque
import estimate_transition
import collect_cartpole_data
import Iterative_Cal_Q
import inference
import two_stage_inference
from scipy.optimize import minimize
import time
import collect_cartpole_data
ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10**6
LEARNING_RATE = 0.001

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 100
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 32

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 0.1
explore = "ep_greedy"
#explore = "random"

class ReplayBuffer():

  def __init__(self, max_size):
    self.max_size = max_size
    self.transitions = deque()

  def add(self, observation, action, reward, observation2):
    if len(self.transitions) > self.max_size:
      self.transitions.popleft()
    self.transitions.append((observation, action, reward, observation2))

  def sample(self, count):
    return random.sample(self.transitions, count)

  def size(self):
    return len(self.transitions)

def get_q(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  return model.predict(np_obs)

def train(model, observations, targets):
  # for i, observation in enumerate(observations):
  #   np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  #   print "t: {}, p: {}".format(model.predict(np_obs),targets[i])
  # exit(0)

  np_obs = np.reshape(observations, [-1, OBSERVATIONS_DIM])
  np_targets = np.reshape(targets, [-1, ACTIONS_DIM])

  model.fit(np_obs, np_targets, epochs=1, verbose=0)

def predict(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  return model.predict(np_obs)

def get_model():
  model = Sequential()
  model.add(Dense(16, input_shape=(OBSERVATIONS_DIM, ), activation='relu'))
  model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
  model.add(Dense(2, activation='linear'))

  model.compile(
    optimizer=Adam(lr=LEARNING_RATE),
    loss='mse',
    metrics=[],
  )

  return model

def update_action(action_model, target_model, sample_transitions):
  random.shuffle(sample_transitions)
  batch_observations = []
  batch_targets = []

  for sample_transition in sample_transitions:
    old_observation, action, reward, observation = sample_transition

    targets = np.reshape(get_q(action_model, old_observation), ACTIONS_DIM)
    targets[action] = reward
    if observation is not None:
      predictions = predict(target_model, observation)
      new_action = np.argmax(predictions)
      targets[action] += GAMMA * predictions[0, new_action]

    batch_observations.append(old_observation)
    batch_targets.append(targets)

  train(action_model, batch_observations, batch_targets)


def train_collect_data(explore, Total_sample, replaysize = REPLAY_MEMORY_SIZE, opt= None):
    #steps_until_reset = TARGET_UPDATE_FREQ
    random_action_probability = INITIAL_RANDOM_ACTION

    # Initialize replay memory D to capacity N
    replay = ReplayBuffer(replaysize)

    # Initialize action-value model with random weights
    action_model = get_model()

    # Initialize target model with same weights
    # target_model = get_model()
    # target_model.set_weights(action_model.get_weights())

    env = gym.make('CartPole-v0')
    # env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
    cur_step = 0
    iteration = 0
    observation = env.reset()
    episode = 0
    # for episode in range(NUM_EPISODES):
    #  observation = env.reset()
    ## training stage
    total_train = 0.
    train_data =[]
    while cur_step < Total_sample:
        old_observation = observation
        # if episode % 10 == 0:
        #   env.render()
        if explore == "ep_greedy":
            random_action_probability *= RANDOM_ACTION_DECAY
            random_action_probability = max(random_action_probability, 0.1)
            if np.random.random() < random_action_probability:
                action = np.random.choice(range(ACTIONS_DIM))
            else:
                q_values = get_q(action_model, observation)
                action = np.argmax(q_values)
        if explore == "random":
            action = np.random.choice(range(ACTIONS_DIM))
        if explore == "Q-OCBA":
            s = collect_cartpole_data.explore_state_less(observation)
            prob =  opt[s * ACTIONS_DIM : (s+1) * ACTIONS_DIM]
            prob =  prob / np.sum(prob)
            action = np.random.choice(range(ACTIONS_DIM), p = prob)
            #print(s, prob, action)
            #exit()

        observation, reward, done, info = env.step(action)
        reward = reward if not done else -200
        cur_step += 1
        iteration += 1
        train_data.append((old_observation, action, reward, observation))

        if done:
            total_train += iteration
            episode += 1
            #print 'Episode {}, iterations: {}'.format(episode, iteration)
            iteration = 0
            observation = env.reset()
            replay.add(old_observation, action, reward, None)

        else:
            replay.add(old_observation, action, reward, observation)
        if replay.size() >= MINIBATCH_SIZE:
            sample_transitions = replay.sample(MINIBATCH_SIZE)
            update_action(action_model, action_model, sample_transitions)
                #steps_until_reset -= 1

        # if steps_until_reset == 0:
        #   target_model.set_weights(action_model.get_weights())
        #   steps_until_reset = TARGET_UPDATE_FREQ
    return total_train, episode, train_data, action_model

def main():
  rep = 10
  eva_train = []
  eva_test = []
  episodes_trained = []
  num_iter = 200
  gamma = 0.99
  dims = (4, 4, 4, 4)
  n_s = np.prod(dims)
  n_a = 2
  Q_0 = np.zeros(n_s * n_a)
  numdata1 = 1000
  print("first stage data number is {}".format(numdata1))
  for i in range(rep):
      total_train, episode, train_data, action_model = train_collect_data("random", numdata1)

      data = collect_cartpole_data.pre_process(train_data, dims)
      # print(c)
      # for d in train_data:
      #    print(d)
      # exit()

      new_r_n, var_r_n, p_n = estimate_transition.find_rmean_rvar_p_estimate(data, n_s, n_a)
      Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, new_r_n, num_iter, gamma, n_s, n_a)
      V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
      #print(new_r_n)
      #print(Q_n)
      dummy_f_n =  np.ones(n_s * n_a)
      I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = inference.get_Sigma_n_comp(p_n, dummy_f_n, var_r_n, V_n,
                                                                                            gamma,
                                                                                            n_s, n_a, V_n_max_index)
      quad_consts = np.zeros((n_s, n_a))
      denom_consts = np.zeros((n_s, n_a, n_s * n_a))

      for i in range(n_s):
          for j in range(n_a):
              if j != V_n_max_index[i]:
                  minus_op = np.zeros(n_s * n_a)
                  minus_op[i * n_a + j] = 1
                  minus_op[i * n_a + V_n_max_index[i]] = -1
                  denom_consts[i][j] = np.power(np.dot(minus_op, I_TM), 2) * np.diag(cov_V_D)
                  quad_consts[i][j] = (Q_n[i * n_a + j] - Q_n[i * n_a + V_n_max_index[i]]) ** 2

      A, b, G, h = two_stage_inference.construct_contrain_matrix(p_n, n_s, n_a)
      AA = np.array(A)
      bb = np.asarray(b)

      def fun(x):
          return x[0]

      def cons(x, i, j):
          z = x[0]
          w = x[1:]
          return quad_consts[i][j] / (np.sum(np.multiply(denom_consts[i][j], np.reciprocal(w)))) - z

      def eqcons(x, a, b):
          return np.dot(a, x[1:]) - b

      constraints = []
      for i in range(n_s):
          for j in range(n_a):
              if j != V_n_max_index[i]:
                  constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: up_c / (
                      np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) + x[0],
                                      'args': (quad_consts[i][j], denom_consts[i][j])})

      for i in range(AA.shape[0]):
          constraints.append({'type': 'eq', 'fun': lambda x, a, b: np.dot(a, x[1:]) - b, 'args': (AA[i], b[i])})
      constraints = tuple(constraints)
      bnds = []
      for i in range(n_s * n_a + 1):
          bnds.append((1e-5, None))
      bnds = tuple(bnds)
      initial = np.ones(n_s * n_a + 1) / (n_s * n_a)

      def func_val(x):
          vals = []
          for i in range(n_s):
              for j in range(n_a):
                  if j != V_n_max_index[i]:
                      vals.append(quad_consts[i][j] / (2 *
                                                       np.sum(np.multiply(denom_consts[i][j], np.reciprocal(x)))))
          z = np.min(vals)
          # print (z)
          # print (vals)
          return z

      initial[0] = 0
      # print(initial)
      t_1 = time.time()
      res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                     constraints=constraints)
      x_opt = res.x[1:]
      runnung_t = time.time() - t_1
      #print("optimization running time is {}".format(runnung_t))
      print("optimal stationary distribution")
      for i in range(n_s):
          prob = x_opt[i* n_a : (i+1)* n_a]
          prob =  prob / np.sum(prob)
          print(prob)
      exit()
      total_train, episode, train_data, action_model = train_collect_data("Q-OCBA", 3000, opt = x_opt)
      #exit()

      ave_train = total_train / episode
      episodes_trained.append(episode)
      # test stage
      total_test = 0.
      env = gym.make('CartPole-v0')
      for episode in range(NUM_EPISODES):
          observation = env.reset()
          for iteration in range(MAX_ITERATIONS):
              q_values = get_q(action_model, observation)
              action = np.argmax(q_values)
              observation, reward, done, info = env.step(action)
              if done:
                  total_test += iteration
                  #print 'Episode {}, iterations: {}'.format(episode,iteration)
                  break
      ave_test = total_test / NUM_EPISODES
      print("{}: average train and average test are {} and {}".format("Q-OCBA", ave_train, ave_test))
      eva_train.append(ave_train)
      eva_test.append(ave_test)
  epi_train_mean =  np.mean(episodes_trained)
  epi_train_std =  np.std(episodes_trained)
  eva_train_mean  = np.mean(eva_train)
  eva_train_std  = np.std(eva_train)
  eva_test_mean  = np.mean(eva_test)
  eva_test_std  = np.std(eva_test)

  print("rep: average train and average test are {} and {}, number of episodes trained mean  is {}".format(eva_train_mean, eva_test_mean, epi_train_mean))
  print("rep: average std train and average test are {} and {}, number of episodes trained std  is {}".format(eva_train_std, eva_test_std, epi_train_std))




if __name__ == "__main__":
  main()