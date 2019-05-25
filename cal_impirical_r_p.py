import numpy as np
import pickle
import collect_data_swimmer
import time

def cal_impirical_stats(data, n_s, n_a):

    num_data = len(data)
    p_n = np.zeros(n_s * n_a * n_s)
    r_n = np.zeros(n_s * n_a)
    var_r_n = np.zeros(n_s * n_a)
    f_n = np.zeros(n_s * n_a)
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
    for i in range(n_s):
        for j in range(n_a):
            if f_n[i * n_a + j]!=0:
                p_n[(i * n_s * n_a + j * n_s) : (i * n_s * n_a + (j+1) * n_s)] = np.divide(p_n[(i * n_s * n_a + j * n_s) : (i * n_s * n_a + (j+1) * n_s)], f_n[i * n_a + j])
                r_n[i * n_a + j] = np.divide(r_n[i * n_a + j], f_n[i * n_a + j])
                var_r_n[i * n_a + j] = np.divide(var_r_n[i * n_a + j], f_n[i * n_a + j]) - (r_n[i * n_a + j])**2
    f_n = np.divide(f_n, num_data )
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