import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import os, argparse
from nn_model import carpole_net_target
import pickle
import collect_data_swimmer
import matplotlib.pyplot as plt


class TrainAgent:
    def __init__(self, sess, model, num_action, restore=False, discount=0.99, lr=1e-3, clip_grads=1.):
        self.sess, self.discount = sess, discount
        self.num_action = num_action

        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.network, self.inputs = model(num_action)

        self.loss_val, self.loss_inputs = self._loss_func()
        self.step = tf.Variable(0, trainable=False)

        #opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        self.train_op = layers.optimize_loss(loss=self.loss_val, optimizer=opt, learning_rate = None, global_step= self.global_step_tensor, clip_gradients=clip_grads)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('weights/Q_nn'))
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/Q_nn', graph=None)
        self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))


    def train(self, states, next_states, actions, rewards, batch_size = 1024, max_iter = 1000):
        num_data = len(states)
        target_val = self.sess.run(self.network, feed_dict={self.inputs: next_states})
        ys = np.array(rewards) + self.discount* np.max(target_val, axis=1)
        iter = 0
        losses = []
        while iter < max_iter:
            if iter%100 == 0:
                print(iter)
            batch_index = np.random.randint(0, num_data, batch_size)
            batch_index = batch_index.astype(int)
            batch_ys = np.array(ys)[batch_index]
            batch_actions = np.array(actions)[batch_index]
            batch_rewards = np.array(rewards)[batch_index]
            batch_states = np.array(states)[batch_index]
            feed_dict = dict(zip(self.loss_inputs, [batch_actions] + [batch_rewards] + [batch_ys]))
            feed_dict[self.inputs] = batch_states
            result, loss = self.sess.run([self.train_op, self.loss_val], feed_dict)
            self.step = self.step + 1
            result_summary, tenboard_step = self.sess.run([self.summary_op, self.step], feed_dict)
            self.summary_writer.add_summary(summarize(Q_est=np.mean(ys)), global_step=tenboard_step)
            self.summary_writer.add_summary(result_summary, tenboard_step)
            losses.append(loss)
            iter +=1
        return losses

    def _loss_func(self):
        returns = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])
        ys = tf.placeholder(tf.float32, [None])
        q_vals = select(actions, self.network)
        value_loss = tf.reduce_mean(tf.square(ys - q_vals))
        tf.summary.scalar('loss/value', value_loss)
        return value_loss, [actions] + [returns]+[ys]
    # state_space should be list of float states
    def get_Q_val(self, state_space):
        Q_vals = self.sess.run(self.network, feed_dict={self.inputs: state_space})
        Q_vals = Q_vals.flatten()
        return Q_vals




def select(acts, value):
    return tf.gather_nd(value, tf.stack([tf.range(tf.shape(value)[0]), acts], axis=1))


def summarize(**kwargs):
    summary = tf.Summary()
    for k, v in kwargs.items():
        summary.value.add(tag=k, simple_value=v)
    return summary



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.reset_default_graph()
    sess = tf.Session()
    num_action = 2
    model = carpole_net_target
    gamma = 0.95
    train_agent = TrainAgent(sess, model, num_action, discount=gamma)
    n_s = 30
    n_a = 2
    num_data = 1000000
    print("num_data is {}".format(num_data))
    s_0 = 2
    # data = pickle.load(open("data", "rb"))
    p = np.zeros(n_s * n_a * n_s)
    r = np.zeros(n_s * n_a)
    r[0] = 0.1
    r[-1] = 10.
    p[0 * n_s * n_a + 0 * n_s + 0] = 1.
    p[0 * n_s * n_a + 1 * n_s + 0] = 0.7
    p[0 * n_s * n_a + 1 * n_s + 1] = 0.3
    for i in range(1, (n_s - 1)):
        p[i * n_a * n_s + 0 * n_s + (i - 1)] = 1
        p[i * n_a * n_s + 1 * n_s + (i - 1)] = 0.1
        p[i * n_a * n_s + 1 * n_s + i] = 0.6
        p[i * n_a * n_s + 1 * n_s + (i + 1)] = 0.3
    p[(n_s - 1) * n_s * n_a + 0 * n_s + (n_s - 2)] = 1
    p[(n_s - 1) * n_s * n_a + 1 * n_s + (n_s - 2)] = 0.7
    p[(n_s - 1) * n_s * n_a + 1 * n_s + (n_s - 1)] = 0.3
    state_space = np.linspace(0, n_s, n_s, endpoint=False)
    states_mean = np.mean(state_space)
    states_std = np.std(state_space)
    state_space = (state_space - states_mean) / states_std
    state_space = np.array([[s] for s in state_space])

    data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop = 0.85, save_data = True)
    data = np.array(data)
    states = data[:, 0]
    states = (states - states_mean )/ states_std
    states = np.array([[s] for s in states])
    actions = data[:, 1]
    rewards = data[:, 2]
    next_states = data[:, 3]
    next_states = (next_states - states_mean) / states_std
    next_states = np.array([[s] for s in next_states])

    train_agent.get_Q_val(state_space)
    exit()
    losses = train_agent.train(states, next_states, actions , rewards, max_iter=1000)
    plt.plot(losses)
    plt.show()
    print("so far so good")


if __name__ == "__main__":
        main()
