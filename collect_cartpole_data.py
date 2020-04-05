
import numpy as np
import gym
import cal_impirical_r_p
import statistics

def cal_absorbed_state(s):
    # s is of dimension 4
    s1, s2, s3, s4 = s
    s1 = np.piecewise(s1, [(s1>=-2.5) & (s1<-2), (s1>=-2) & (s1<-1.5), (s1>=-1.5) & (s1<-1), (s1>=-1) & (s1<-0.5), (s1>=-0.5) & (s1<0), (s1>=0) & (s1<0.5), (s1>=0.5) & (s1<1), (s1>=1) & (s1<1.5), (s1>=1.5) & (s1<2), (s1>=2) & (s1<2.5)],
                      [0,1,2,3,4,5,6,7,8,9])
    s2 = np.piecewise(s2, [(s2 < -1.5), (s2 >= -1.5) & (s2< -1),
                           (s2 >= -1) & (s2 < -0.5), (s2 >= -0.5) & (s2 < 0), (s2 >= 0) & (s2 < 0.5), (s2 >= 0.5) & (s2 < 1),
                           (s2 >= 1) & (s2 < 1.5), (s2 >= 1.5)], [0, 1, 2, 3, 4, 5, 6, 7])
    s3 = np.piecewise(s3, [(s3 < -40), (s3 >= -40) & (s3 < -35), (s3 >= -35) & (s3 < -30), (s3 >= -30) & (s3 < -25),
                           (s3 >= -25) & (s3 < -20), (s3 >= -20) & (s3 < -15), (s3 >= -15) & (s3 < -10), (s3 >= -10) & (s3 < -5),
                           (s3 >= -5) & (s3 < 0), (s3 >= 0) & (s3 < 5), (s3 >= 5) & (s3 < 10), (s3 >= 10) & (s3 < 15), (s3 >= 15) & (s3 < 20), (s3 >= 20) & (s3 < 25), (s3 >= 25) & (s3 < 30), (s3 >= 30) & (s3 < 35), (s3 >= 35) & (s3 < 40), s3 >= 40 ],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    s4 = np.piecewise(s4, [(s4 < -1.5), (s4 >= -1.5) & (s4 < -1),
                           (s4 >= -1) & (s4 < -0.5), (s4 >= -0.5) & (s4 < 0), (s4 >= 0) & (s4 < 0.5), (s4 >= 0.5) & (s4 < 1),
                           (s4 >= 1) & (s4 < 1.5), s4 >= 1.5], [0, 1, 2, 3, 4, 5, 6, 7])

    s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
    return s1, s2, s3, s4
def explore_state(s, dims =(10, 8, 18, 8)):
    s1, s2, s3, s4 = cal_absorbed_state(s)
    state = s1 * dims[1] * dims[2] * dims[3] + s2 * dims[2] * dims[3] + s3 * dims[3] + s4
    return state

def cal_absorbed_state_less(s):
    # s is of dimension 4
    s1, s2, s3, s4 = s
    #s1 = np.piecewise(s1, [s1<-0.2, (s1>=-0.2) & (s1<-0.1), (s1>=-0.1) & (s1<-0), (s1>=0) & (s1<0.1), (s1>=0.1) & (s1<0.2), s1>=0.2],
    #                  [0,1,2,3,4,5])
    s1 = np.piecewise(s1, [s1 < -0.1,  (s1 >= -0.1) & (s1 < 0), (s1 >= 0) & (s1 < 0.1), s1 >= 0.1],
                      [0, 1, 2, 3])
    s2 = np.piecewise(s2, [s2< -1,
                           (s2 >= -1) & (s2 < 0), (s2 >= 0) & (s2 < 1), (s2 >= 1)], [0, 1, 2, 3])
    s3 = np.piecewise(s3, [(s3 < -0.1),(s3 >= -0.1) & (s3 < 0), (s3 >= 0) & (s3 < 0.1),  s3 >= 0.1 ],
                      [0, 1, 2, 3])
    s4 = np.piecewise(s4, [(s4 < -1), (s4 >= -1) & (s4 < 0),
                           (s4 >= 0) & (s4 < 1), s4 >= 1], [0, 1, 2, 3])

    s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
    return s1, s2, s3, s4
def explore_state_less(s, dims =(4, 4, 4, 4)):
    s1, s2, s3, s4 = cal_absorbed_state_less(s)
    state = s1 * dims[1] * dims[2] * dims[3] + s2 * dims[2] * dims[3] + s3 * dims[3] + s4
    return state
def pre_process(data, dims):
    ans = []
    for s,a,r, snext in data:
        s = explore_state_less(s, dims)
        snext = explore_state_less(snext, dims = dims)
        ans.append((s,a,r,snext))
    return ans





def main():
    ENV_NAME = "CartPole-v0"
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    train_data = []
    num_train_data = 30000
    #print(observation_space, action_space)
    #exit()
    state = env.reset()
    c = 0
    while len(train_data) < num_train_data:
        action =np.random.randint(2)
        #action =1
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -200
        #state_next = np.reshape(state_next, [1, observation_space])
        #dqn_solver.remember(state, action, reward, state_next, terminal)
        train_data.append((state, action, reward, state_next))
        #print(state, action, reward, state_next)
        state = state_next
        if terminal:
            c +=1
            state = env.reset()
    dims = (4,4,4,4)
    data = pre_process(train_data, dims)
    #print(c)
    #for d in train_data:
    #    print(d)
    #exit()
    n_s = np.prod(dims)
    n_a = 2
    p_n, r_n, f_n, var_r_n = cal_impirical_r_p. cal_impirical_stats(data, n_s, n_a)
    print(len(f_n))
    nonzero_freq = [f for f in f_n if f!=0]
    print(len(nonzero_freq))
    print(max(f_n), min(nonzero_freq), statistics.median(nonzero_freq))

if __name__ == "__main__":
    main()