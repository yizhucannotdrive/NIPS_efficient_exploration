import numpy as np
import Iterative_Cal_Q
import inference
import collect_data_swimmer
import optimize_pfs
import argparse
import cal_impirical_r_p


class parameter_prior:

    def __init__ (self, n_s=0, n_a=0, s_0 = 2):
        self.n_s = n_s
        self.n_a = n_a
        self.pprior = np.array([1.] * self.n_s * (self.n_s * self.n_a))
        self.r_mean = np.ones(self.n_s * self.n_a)
        self.r_std = np.ones(self.n_s * self.n_a)
        #self.r_std = np.zeros(self.n_s * self.n_a)
        self.s_0  = s_0

    def sampled_MDP_Q(self, Q_0,  num_iter, gamma):
        transition = []
        for i in range(self.n_s):
            for j in range(self.n_a):
                transition += list(np.random.dirichlet(self.pprior[(i*self.n_s* self.n_a + j * self.n_s): (i*self.n_s* self.n_a + j * self.n_s + self.n_s)]))
        transition = np.array(transition)
        rewards = np.array([ np.random.normal(self.r_mean[i], self.r_std[i]) for i in range(self.n_s * self.n_a)])
        Q_estimate = Iterative_Cal_Q.cal_Q_val(transition, Q_0, rewards, num_iter , gamma, self.n_s, self. n_a)
        #print(Q_estimate)
        return Q_estimate

    ## take Q-value to data collection

    def update(self, data, r_sigma=1.0):
        for d in data:
            s, a, r, s_1 = d
            self.pprior[s * self.n_s * self.n_a + a * self.n_s + s_1] += 1
            std_prior = self.r_std[self.n_a * s + a]
            # print(std_prior, r_sigma)
            denom = 1. / (std_prior * std_prior) + 1. / (r_sigma * r_sigma) if r_sigma != 0 else 0
            self.r_mean[s * self.n_a + a] = ((1. / (std_prior * std_prior) * self.r_mean[s * self.n_a + a] + r / (
                        r_sigma * r_sigma)) / denom) if denom != 0 else r
            self.r_std[s * self.n_a + a] = (np.sqrt(1. / denom)) if denom != 0 else 0
            # print(s, a, self.r_mean, self.r_std)
            # exit()
            self.s_0 = s_1
        #print(self.pprior)
        #print(self.r_mean)
        #print(self.r_std)






#python PSPE.py --rep 100 --epi_step_num 100 --r0 2.0 --numdata 1000 --rstd 0.0 --two_stage True --rightprop 0.6 --beta 0.25
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=0.0, help='value of r0')
    #parser.add_argument('--r_prior', nargs="?", type=float, default=1.0, help='prior value of reward function')
    parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--epi_step_num', nargs="?", type=int, default=100, help='number of episode steps')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')
    parser.add_argument('--beta', nargs="?", type=float, default=0.25,
                        help='beta')
    parser.add_argument('--two_stage', nargs="?", type=bool, default=True,
                        help='if run two stage or sequential experiment')
    args = parser.parse_args()
    print("PSPE")
    num_iter, gamma, n_s, n_a, num_rep = 200, 0.95, 5, 2, args.rep
    right_prop = args.rightprop

    Q_0 = np.zeros(n_s * n_a)
    V_0 = np.zeros(n_s)
    p = np.zeros(n_s * n_a * n_s)
    r = np.zeros(n_s * n_a)
    r[0] = args.r0
    r[-1] = 10.
    r_std = args.rstd
    print("reward standard deviation is {}".format(r_std))
    # r[0] = 10.
    # r[-1] = 0.1
    print("r[0] and r[-1] are {}, {}".format(r[0], r[-1]))
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
    Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    V_real, V_max_index = inference.get_V_from_Q(Q_real, n_s, n_a)
    #print("Q real is {}".format(Q_real))
    s_0 = 2

    ## PSPE
    if not args.two_stage:
        print("sequential implementation")
        Total_data = args.numdata
        print("total num of data is {}".format(Total_data))
        episode_steps = args.epi_step_num
        numdata_1 = episode_steps
        numdata_2 = Total_data - numdata_1
        print("epsisode timestep is {}".format(episode_steps))
        num_datas = [episode_steps] * (numdata_2 / episode_steps)
    else:
        print("two_stage implementation")
        Total_data = args.numdata
        print("total num of data is {}".format(Total_data))
        numdata_1 = Total_data * 3 / 10
        numdata_2 = Total_data - numdata_1
        episodes = 100
        num_datas = [numdata_2 / episodes] * episodes

    CS_num = 0.
    beta  = args.beta
    rou = np.ones(n_s) / n_s
    future_V = np.zeros(num_rep)
    for i in range(num_rep):
        para_cl = parameter_prior(n_s,n_a, s_0)
        while True:
            data1 = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_std)
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data1, n_s, n_a)
            Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
            # print("first stage visiting frequency is {}".format(f_n))
            if f_n.all() != 0:
                break
        para_cl.update(data1, r_sigma= r_std)
        Q_estimate = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
        #print(Q_estimate)

        for num_data in num_datas:
            data = collect_data_swimmer.collect_data(p, r, num_data, para_cl.s_0, n_s, n_a, Q= Q_estimate, epsilon= 0, std = r_std)
            para_cl.update(data, r_sigma= r_std)
            Q_estimate_1 = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
            V_n_1, V_n_max_index_1 = inference.get_V_from_Q(Q_estimate_1, n_s, n_a)
            sim = np.random.binomial(1,beta,1)[0]
            if sim:
                Q_estimate = Q_estimate_1
            else:
                while True:
                    Q_estimate_2 = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
                    V_n_2, V_n_max_index_2 = inference.get_V_from_Q(Q_estimate_2, n_s, n_a)
                    if V_n_max_index_2 != V_n_max_index_1:
                        break
                Q_estimate =  Q_estimate_2
        #print(Q_estimate)
        #print(para_cl.pprior)
        #print(para_cl.r_mean)
        V_here = optimize_pfs.policy_val_iteration(Q_estimate, n_s, n_a, V_0, num_iter, r, p, gamma)
        # print(V_here, V_real)
        future_V[i] = np.dot(rou, V_here)
        FS_bool = optimize_pfs.FS_bool(Q_estimate, V_max_index, n_s, n_a)
        CS_num += FS_bool
    PCS = np.float(CS_num) / num_rep
    CI_len = 1.96 * np.sqrt(PCS * (1 - PCS) / num_rep)
    fv = np.mean(future_V)
    fv_std = np.std(future_V)
    rv = np.dot(rou, V_real)
    diff = rv - fv
    # print(CS_num_naive)
    print("PCS is {}, with CI length {}".format(PCS, CI_len))
    print("future value func is {} with  CI length {}, real value is {}, diff is {}".format(fv, 1.96 * fv_std / np.sqrt(
        num_rep), rv, diff))






if __name__=="__main__":
    main()