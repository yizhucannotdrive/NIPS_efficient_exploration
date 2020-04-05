import numpy as np
import pickle
import collect_data_swimmer
import time
import collections

def dirichlet_process(data, p_n, n_s, n_a):
    num_data = len(data)
    train_set = set()
    for i in range(num_data):
        data_i = data[i]
        s = int(data_i[0])
        a = int(data_i[1])
        s_next = int(data_i[3])
        p_n[s * n_s * n_a + a *n_s + s_next] +=1
        train_set.add((s,a))
    for i in range(n_s):
        for j in range(n_a):
            p_n[i * n_s * n_a + j *n_s : i * n_s * n_a + j *n_s + n_s ] /= np.sum(p_n[i * n_s * n_a + j *n_s : i * n_s * n_a + j *n_s + n_s ])
    return p_n, train_set


def reward_process(data, n_s, n_a):
    num_data = len(data)
    eval_set = set()
    rew= np.zeros(n_s * n_a)
    f = np.zeros(n_s * n_a)
    for i in range(num_data):
        data_i = data[i]
        s = int(data_i[0])
        a = int(data_i[1])
        r = int(data_i[2])
        rew[s*n_a +a] +=r
        f[s*n_a +a] +=1
        eval_set.add((s, a))
    for s, a in eval_set:
        rew[s * n_a +a] /= f[s * n_a + a]
    return rew, f, eval_set

def reward_process_var(data, n_s, n_a):
    num_data = len(data)
    rew_var= collections.defaultdict(list)
    rew_var_new = np.zeros(n_s * n_a)
    f = np.zeros(n_s * n_a)
    for i in range(num_data):
        data_i = data[i]
        s = int(data_i[0])
        a = int(data_i[1])
        r = int(data_i[2])
        rew_var[(s,a)].append(r)
        f[s * n_a + a] += 1
    for s, a in rew_var.keys():
        rew_var_new[s *n_a +a] = np.std(rew_var[(s,a)])
    return rew_var_new,f, rew_var.keys()


def cal_impirical_stats_approximation(data, n_s, n_a, f_n_def = 0):
    num_data = len(data)
    p_n = np.ones(n_s * n_a * n_s) * 1./n_s
    r_n = np.zeros(n_s * n_a)
    var_r_n = np.zeros(n_s * n_a)
    p_n = dirichlet_process(data, p_n, n_s, n_a)


def cal_impirical_stats(data, n_s, n_a, f_n_def = 0):

    p_n = np.array([0.] * n_s *( n_s * n_a))
    r_n = np.zeros(n_s * n_a)
    var_r_n = np.zeros(n_s * n_a)
    f_n = np.ones(n_s * n_a) * f_n_def
    num_data = len(data)
    for i in range(num_data):
        data_i = data[i]
        s = int(data_i[0])
        a = int(data_i[1])
        r = data_i[2]
        s_next = int(data_i[3])
        p_n[s * n_s * n_a + a *n_s + s_next] +=1
        r_n[s * n_a + a] += r
        var_r_n[s * n_a + a] += r**2
        f_n[s * n_a + a] += 1
    #print("data category finished")
    #print(p_n)
    #print(f_n)
    for i in range(n_s):
        for j in range(n_a):
            if f_n[i * n_a + j]!=0:
                #print(p_n)
                tmp = np.divide(np.array(p_n[(i * n_s * n_a + j * n_s) : (i * n_s * n_a + (j+1) * n_s)]), float(f_n[i * n_a + j]))
                #print(tmp)
                p_n[(i * n_s * n_a + j * n_s) : (i * n_s * n_a + (j+1) * n_s)] = tmp
                #print(p_n)
                r_n[i * n_a + j] = np.divide(r_n[i * n_a + j], f_n[i * n_a + j])
                var_r_n[i * n_a + j] = np.divide(var_r_n[i * n_a + j], f_n[i * n_a + j]) - (r_n[i * n_a + j])**2
    f_n = np.divide(f_n, num_data )
    #print(p_n)
    #print(f_n)
    #exit()
    return p_n, r_n, f_n, var_r_n
def main():
    time_start = time.time()
    n_s = 100
    n_a = 2
    num_data = 100000000
    print("num_data is {}".format(num_data))
    s_0 = 2
    #data = pickle.load(open("data", "rb"))
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
    data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop = 0.85, save_data = True)
    time_data_collect = time.time()
    print("data collection finished using {}  seconds".format(time_data_collect- time_start) )
    data = pickle.load(open("data", "rb"))
    time_load = time.time()
    print("data load finished using {}  seconds".format(time_load- time_data_collect) )
    batch_size = 100000
    batch_index= np.random.randint(0,num_data, batch_size)
    batch_index =  batch_index.astype(int)
    batch_data = np.array(data)[batch_index]
    time_batch = time.time()
    print("data batching finished using {}  seconds".format(time_batch - time_load))
    p_n, r_n, f_n, var_r_n  = cal_impirical_stats(batch_data, n_s, n_a)
    time_cal = time.time()
    print("data stats calculation finished using {}  seconds".format(time_cal - time_batch))
    print(p_n, r_n, f_n)






if __name__ == "__main__":
    main()