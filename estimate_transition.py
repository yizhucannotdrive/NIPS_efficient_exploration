from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation
from keras.optimizers import Adam
from keras.activations import softmax
from  keras import backend as K
import numpy as np
import random
import gym
import collect_cartpole_data
import cal_impirical_r_p

LEARNING_RATE = 0.001
ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MINIBATCH_SIZE = 32
STATES = [[-0.1, -0.05, 0.05, 0.1], [-1, -0.5, 0.5, 1], [-0.1, -0.05, 0.05, 0.1], [-1, -0.5, 0.5, 1]]
def softMaxAxis1(x, axis = 2):
    ndim = K.ndim(x)
    if ndim >= 2:
        e = K.exp(x)
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)
def predict(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  return model.predict(np_obs)

def get_model():
    model = Sequential()
    model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dense(32, input_shape=(16,), activation='relu'))
    model.add(Dense(64, input_shape=(16,), activation='relu'))
    model.add(Dense(256, input_shape=(16,), activation='relu'))
    model.add(Dense(ACTIONS_DIM * (OBSERVATIONS_DIM ** 4), activation='linear'))
    model.add(Reshape((ACTIONS_DIM, OBSERVATIONS_DIM **4)))
    model.add(Activation(softMaxAxis1))
    #model.summary()
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mse',
        metrics=[],
    )
    return model


def get_model_r():
    model = Sequential()
    model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dense(32, input_shape=(16,), activation='relu'))
    model.add(Dense(64, input_shape=(16,), activation='relu'))
    model.add(Dense(16, input_shape=(16,), activation='relu'))
    model.add(Dense(ACTIONS_DIM, activation='linear'))
    #model.summary()
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='mse',
        metrics=[],
    )
    return model


def get_model_r_bino():
    model = Sequential()
    model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    model.add(Dense(32, input_shape=(16,), activation='relu'))
    model.add(Dense(64, input_shape=(16,), activation='relu'))
    model.add(Dense(16, input_shape=(16,), activation='relu'))
    model.add(Dense(ACTIONS_DIM, activation='sigmoid'))
    #model.add(Dense(ACTIONS_DIM *2,  activation='linear'))
    #model.add(Reshape((ACTIONS_DIM, 2)))
    #model.add(Activation('sigmoid'))
    #model.summary()
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='binary_crossentropy',
        #loss='mse',
        metrics=['accuracy'],
    )
    return model

def get_internal_state(s):
    obs = []
    for i in range(4):
        obs.append(s / (4 ** (3 - i)))
        #obs.append(STATES[i][s / (4 ** (3 - i))])
        s = s % (4 ** (3 - i))
    return obs

def estimate_transition(p_n, trainsets, n_s = OBSERVATIONS_DIM, n_a = ACTIONS_DIM, num_epoch = 1000):
    model =  get_model()
    traintransitions = []
    for s,a in trainsets:
        obs = get_internal_state(s)
        traintransitions.append((obs, s, a))
    for i in range(num_epoch):
        random.shuffle(traintransitions)
        batchsample = traintransitions[:MINIBATCH_SIZE]
        samples = []
        targets = []
        for sample in batchsample:
            state,s, a = sample
            target = np.reshape(predict(model, state), (ACTIONS_DIM, -1))
            target[a] = p_n[s * n_s * n_a + a *n_s : s * n_s * n_a + a *n_s + n_s ]
            samples.append(state)
            targets.append(target)
        samples = np.reshape(samples, [-1, OBSERVATIONS_DIM])
        targets =  np.reshape(targets, [-1, ACTIONS_DIM, OBSERVATIONS_DIM**4])
        model.fit(samples, targets, epochs=1, verbose=0)
        pred = predict(model, samples[0])
    new_p_n = np.ones(n_s * n_a * n_s) * 1./n_s
    for i in range(n_s):
        new_p_n[i * n_s * n_a : (i+1) * n_s * n_a ] = np.reshape(predict(model, get_internal_state(i)), -1)
    return new_p_n


def estimate_rewards_var(v_var_n, trainsets, n_s = OBSERVATIONS_DIM, n_a = ACTIONS_DIM, num_epoch = 1000):
    model =  get_model_r()
    trainvars = []
    for s,a in trainsets:
        obs = get_internal_state(s)
        trainvars.append((obs, s, a))
    for i in range(num_epoch):
        random.shuffle(trainvars)
        batchsample = trainvars[:MINIBATCH_SIZE]
        samples = []
        targets = []
        for sample in batchsample:
            state,s, a = sample
            target = np.reshape(predict(model, state), (ACTIONS_DIM, -1))
            target[a] = v_var_n[s * n_a + a]
            samples.append(state)
            targets.append(target)
        samples = np.reshape(samples, [-1, OBSERVATIONS_DIM])
        targets =  np.reshape(targets, [-1, ACTIONS_DIM])
        model.fit(samples, targets, epochs=1, verbose=0)
        pred = predict(model, samples[0])
    new_r_var = np.ones(n_s * n_a)
    for i in range(n_s):
        new_r_var[i * n_a : (i+1)  * n_a ] = np.reshape(predict(model, get_internal_state(i)), -1)
    return new_r_var



def estimate_reward(data, n_s = OBSERVATIONS_DIM, n_a = ACTIONS_DIM, num_epoch = 1000):
    model =  get_model_r()
    for i in range(num_epoch):
        random.shuffle(data)
        batchsample = data[:MINIBATCH_SIZE]
        samples = []
        targets = []
        for sample in batchsample:
            s, a, r, _  = sample
            obs = get_internal_state(s)
            samples.append(obs)
            target = np.reshape(predict(model, obs), (ACTIONS_DIM, -1))
            target[a] = r
            targets.append(target)
        samples = np.reshape(samples, [-1, OBSERVATIONS_DIM])

        targets = np.reshape(targets, [-1, ACTIONS_DIM])
        model.fit(samples, targets, epochs=1, verbose=0)
        pred = predict(model, samples[0])
    new_r_n = np.ones(n_s * n_a)
    for i in range(n_s):
        #print(get_internal_state(i))
        #print(predict(model, get_internal_state(i)))
        new_r_n[i * n_a : (i+1) * n_a ] = np.reshape(predict(model, get_internal_state(i)), -1)
    return new_r_n



def estimate_reward_bino(data, n_s = OBSERVATIONS_DIM, n_a = ACTIONS_DIM, num_epoch = 1000):
    model =  get_model_r_bino()
    for i in range(num_epoch):
        random.shuffle(data)
        batchsample = data[:MINIBATCH_SIZE]
        samples = []
        targets = []
        for sample in batchsample:
            s, a, r, _  = sample
            obs = get_internal_state(s)
            samples.append(obs)
            target = np.reshape(predict(model, obs), (ACTIONS_DIM, -1))
            #target[a] = np.array([1,0]) if r==1 else np.array([0,1])
            target[a] = 1 if r==1 else 0
            #print(target)
            #exit()
            targets.append(target)
        samples = np.reshape(samples, [-1, OBSERVATIONS_DIM])

        targets = np.reshape(targets, [-1, ACTIONS_DIM])
        model.fit(samples, targets, epochs=1, verbose=0)
    pred = predict(model, samples[0:100])
    #print(pred)
    #exit()
    new_r_n = np.ones(n_s * n_a)
    for i in range(n_s):
        #print(get_internal_state(i))
        #print(predict(model, get_internal_state(i)))
        prob_vec = np.reshape(predict(model, get_internal_state(i)), -1)
        new_r_n[i * n_a : (i+1) * n_a ] = prob_vec *1 + (-200)*(1- prob_vec)
    return new_r_n

def estimate_reward_2(rew, eval_set , n_s = OBSERVATIONS_DIM, n_a = ACTIONS_DIM, num_epoch = 1000):
    model =  get_model_r()
    rews = []
    for s,a in eval_set:
        obs = get_internal_state(s)
        rews.append((obs, s, a))
    for i in range(num_epoch):
        random.shuffle(rews)
        batchsample = rews[:MINIBATCH_SIZE]
        samples = []
        targets = []
        for sample in batchsample:
            state,s, a = sample
            target = np.reshape(predict(model, state), (ACTIONS_DIM, -1))
            target[a] = rew[s * n_a +a]
            samples.append(state)
            targets.append(target)
        samples = np.reshape(samples, [-1, OBSERVATIONS_DIM])
        targets =  np.reshape(targets, [-1, ACTIONS_DIM])
        model.fit(samples, targets, epochs=1, verbose=0)
        pred = predict(model, samples[0])
    new_r_n = np.ones(n_s * n_a)
    for i in range(n_s):
        new_r_n[i * n_a : (i+1) * n_a ] = np.reshape(predict(model, get_internal_state(i)), -1)
    return new_r_n


def find_rmean_rvar_p_estimate(data, n_s, n_a):
    # approximate rewards variance
    print("start train var")
    r_var, f, trainset = cal_impirical_r_p.reward_process_var(data, n_s, n_a)
    newr_var = estimate_rewards_var(r_var, trainset, n_s=n_s) ** 2
    #for s, a in trainset:
    #    print(s, a, f[s * n_a + a], r_var[s * n_a + a] ** 2, newr_var[s * n_a + a])
    # print(len(eval_set))
    #exit()

    # approximate rewards
    print("start train r")
    new_r_n = estimate_reward_bino(data, n_s, n_a)
    rew, f, eval_set = cal_impirical_r_p.reward_process(data, n_s, n_a)
    # new_r_n =  estimate_reward_2(rew, eval_set, n_s, n_a)

    #for s, a in eval_set:
    #    print(s, a, f[s * n_a + a], rew[s * n_a + a], new_r_n[s * n_a + a])
    # print(len(eval_set))
    #exit()

    # approximate p_n
    print("start train p")
    p_n = np.ones(n_s * n_a * n_s) * 1. / n_s
    p_n, train_set = cal_impirical_r_p.dirichlet_process(data, p_n, n_s, n_a)
    new_p_n = estimate_transition(p_n, train_set, n_s=n_s)
    #print("output")
    # print(len(train_set))
    # exit()
    #for s, a in train_set:
    #    print(max(p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]),
    #          np.argmax(p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]),
    #          max(new_p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]),
    #          np.argmax(new_p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]))
    #    print(max(p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]) - max(
    #        new_p_n[s * n_s * n_a + a * n_s:  s * n_s * n_a + a * n_s + n_s]))
    return new_r_n, newr_var, new_p_n


def main():
    ENV_NAME = "CartPole-v0"
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    train_data = []
    num_train_data = 1000
    # print(observation_space, action_space)
    # exit()
    state = env.reset()
    c = 0
    while len(train_data) < num_train_data:
        action = np.random.randint(2)
        # action =1
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -200
        # state_next = np.reshape(state_next, [1, observation_space])
        # dqn_solver.remember(state, action, reward, state_next, terminal)
        train_data.append((state, action, reward, state_next))
        # print(state, action, reward, state_next)
        state = state_next
        if terminal:
            c += 1
            state = env.reset()
    dims = (4, 4, 4, 4)
    data = collect_cartpole_data.pre_process(train_data, dims)
    #print(c)
    #for d in train_data:
    #    print(d)
    #exit()
    n_s = np.prod(dims)
    n_a = 2
    r, rvar, p = find_rmean_rvar_p_estimate(data, n_s, n_a)
    print(r, rvar, p)



if __name__ == "__main__":
    main()