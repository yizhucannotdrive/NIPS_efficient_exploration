import numpy as np
import matplotlib.pyplot as plt
import train_agent
import tensorflow as tf
import os
from nn_model import carpole_net_target
import collect_data_swimmer

def cal_Q_val(p,Q_init,r, num_iter, gamma, n_s, n_a, plot_diff = False):
    Q_current = np.copy(Q_init)
    Q_pre = np.copy(Q_init)
    diffs = []
    for t in range(num_iter):
        for i in range(n_s):
            for j in range(n_a):
                exp_next_q = 0
                for k in range(n_s):
                    exp_next_q += p[i * n_s * n_a + j * n_s + k] * np.max(Q_pre[k*n_a: (k+1)*n_a])
                Q_current[i*n_a + j] = r[i*n_a + j] + gamma * exp_next_q
        #print("estimate of {} th iteration is {}".format(t,Q_current))
        diff = np.max(np.abs(Q_current - Q_pre))
        diffs.append(diff)
        Q_pre = np.copy(Q_current)
    if plot_diff:
        plt.plot(diffs)
        plt.show()
    return Q_current

def find_nn_for_linspace_integers(n_s, S_0, space = 10):
    nearest_n_mapping = []
    for i in range(n_s):
        quo = i // space
        rem = i % space
        nn = S_0[quo]  if rem <= space/2 else  S_0[quo+1]
        nearest_n_mapping.append(nn)
    return nearest_n_mapping

def cal_Q_val_approx_nearest_neighbor(p,Q_init,r, num_iter, gamma, S_0, n_s, n_a, plot_diff = False):
    Q_current = np.copy(Q_init)
    Q_pre = np.copy(Q_init)
    #specify how to find nearest neighbor
    nn_ls = find_nn_for_linspace_integers(n_s, S_0)
    diffs = []
    for t in range(num_iter):
        for i in S_0:
            for j in range(n_a):
                exp_next_q = 0
                for k in range(n_s):
                    exp_next_q += p[i * n_s * n_a + j * n_s + k] * np.max(Q_pre[k*n_a: (k+1)*n_a])
                Q_current[i*n_a + j] = r[i*n_a + j] + gamma * exp_next_q
        for i in range(n_s):
            for j in range(n_a):
                Q_current[i * n_a + j] = Q_current[nn_ls[i] * n_a + j]
        diff = np.max(np.abs(Q_current - Q_pre))
        diffs.append(diff)
        Q_pre = np.copy(Q_current)
        #print("estimate of {} th iteration is {}".format(t,Q_current))
    if plot_diff:
        plt.plot(diffs)
        plt.show()
    return Q_current


def get_slopes(Q_current, S_0, n_a):
    slopes  = []
    for i in range(len(S_0)-1):
        slops = []
        for j in range(n_a):
            slope = (Q_current[S_0[i+1]*n_a + j] - Q_current[S_0[i]*n_a + j]) / (S_0[i+1] - S_0[i])
            slops.append(slope)
        slopes.append(slops)
    return slopes

# assume S_0 is already ordered increasingly
def cal_Q_val_approx_linear_interpolation(p,Q_init,r, num_iter, gamma, S_0, n_s, n_a, plot_diff = False):
    Q_current = np.copy(Q_init)
    Q_pre = np.copy(Q_init)
    #specify how to find nearest neighbor
    diffs = []
    for t in range(num_iter):
        for i in S_0:
            for j in range(n_a):
                exp_next_q = 0
                for k in range(n_s):
                    exp_next_q += p[i * n_s * n_a + j * n_s + k] * np.max(Q_pre[k*n_a: (k+1)*n_a])
                Q_current[i*n_a + j] = r[i*n_a + j] + gamma * exp_next_q
        slopes = get_slopes(Q_current, S_0, n_a)
        for i in range(len(S_0)-1):
            for s in range(S_0[i], S_0[i+1]):
                for j in range(n_a):
                    Q_current[s * n_a + j] = Q_current[S_0[i] * n_a + j] + slopes[i][j] * (s-S_0[i])
        diff = np.max(np.abs(Q_current - Q_pre))
        diffs.append(diff)
        Q_pre = np.copy(Q_current)
        #print("estimate of {} th iteration is {}".format(t,Q_current))
    if plot_diff:
        plt.plot(diffs)
    return Q_current


def cal_Q_val_approx_neuralnetwork(num_data, p, r,  num_iter, gamma, s_0, n_s, n_a, max_iter_train, plot_diff = False):
    diffs = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    tf.reset_default_graph()
    sess = tf.Session()
    num_action = 2
    model = carpole_net_target
    trainagent = train_agent.TrainAgent(sess, model, num_action, discount = gamma)
    state_space = np.linspace(0, n_s, n_s, endpoint=False)
    states_mean = np.mean(state_space)
    states_std = np.std(state_space)
    state_space = (state_space - states_mean) / states_std
    state_space = np.array([[s] for s in state_space])
    data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop=0.85, save_data=True)
    data = np.array(data)
    states = data[:, 0]
    states = (states - states_mean) / states_std
    states = np.array([[s] for s in states])
    actions = data[:, 1]
    rewards = data[:, 2]
    next_states = data[:, 3]
    next_states = (next_states - states_mean) / states_std
    next_states = np.array([[s] for s in next_states])
    Q_current = trainagent.get_Q_val(state_space)
    Q_pre = np.copy(Q_current)
    for t in range(num_iter):
        print("current t is {}".format(t))
        _ = trainagent.train(states, next_states, actions, rewards, max_iter = max_iter_train)
        Q_current = trainagent.get_Q_val(state_space)
        diff = np.max(np.abs(Q_current - Q_pre))
        diffs.append(diff)
        Q_pre = np.copy(Q_current)
        print("estimate of {} th iteration is {}".format(t,Q_current))
    if plot_diff:
        plt.plot(diffs)
    return Q_current


def main():
    #approximation = "linear_interpolation"
    approximation = None
    #approximation = "neural_network"
    print("approximation is {}".format(approximation))
    plot_diff = True
    num_iter = 150
    gamma = 0.95
    n_s = 30
    n_s_0 = 11
    S_0 = np.linspace(0, 100, 11)
    S_0[-1] = 99
    S_0 = S_0.astype(int)
    n_a = 2
    p = np.zeros(n_s * n_a * n_s)
    Q_0 = np.zeros(n_s * n_a)
    r = np.zeros(n_s * n_a)
    r[0] = 1
    r[-1] = 10
    p[0 * n_s * n_a + 0 * n_s + 0] = 1.
    p[0 * n_s * n_a  + 1 * n_s + 0] = 0.7
    p[0 * n_s * n_a  + 1 * n_s + 1] = 0.3
    for i in range(1, (n_s-1)):
        p[i* n_a * n_s + 0 *n_s + (i-1)] = 1
        p[i* n_a * n_s + 1 *n_s + (i-1)] = 0.1
        p[i * n_a * n_s + 1 * n_s + i] = 0.6
        p[i * n_a * n_s + 1 * n_s + (i + 1)] = 0.3
    p[(n_s-1) * n_s * n_a + 0 * n_s +  (n_s-2) ] = 1
    p[(n_s - 1) * n_s * n_a + 1 * n_s + (n_s - 2)] = 0.7
    p[(n_s - 1) * n_s * n_a + 1 * n_s + (n_s - 1)] = 0.3
    #print(p)
    #print(r)
    num_data = 100000
    s_0 = 2
    max_iter_train = 10000
    if approximation == None:
        Q_estimate = cal_Q_val(p,Q_0,r, num_iter, gamma, n_s, n_a, plot_diff= plot_diff)
    if approximation == "1nn":
        Q_estimate = cal_Q_val_approx_nearest_neighbor(p,Q_0,r, num_iter, gamma, S_0, n_s, n_a, plot_diff= plot_diff)
    if approximation == "linear_interpolation":
        Q_estimate = cal_Q_val_approx_linear_interpolation(p, Q_0, r, num_iter, gamma, S_0, n_s, n_a, plot_diff= plot_diff)
    if approximation == "neural_network":
        Q_real = cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a, plot_diff=False)
        print("Q_real is {}".format(Q_real))
        Q_estimate = cal_Q_val_approx_neuralnetwork(num_data, p, r,  num_iter, gamma, s_0, n_s, n_a, max_iter_train = max_iter_train, plot_diff = plot_diff)
        print("Q_estimate with neural_nets is {}".format(Q_estimate))

    print(Q_estimate)





if __name__ == "__main__":
    main()