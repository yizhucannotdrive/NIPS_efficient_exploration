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
        #self.pprior = np.array([1./self.n_s]*self.n_s *( self.n_s * self.n_a))
        self.pprior = np.array([1./n_s]*self.n_s *( self.n_s * self.n_a))
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

    def update(self, data, r_sigma = 1.0):
        for d in data:
            s,a, r, s_1 = d
            self.pprior[s *self.n_s* self.n_a + a * self.n_s + s_1] +=1
            std_prior = self.r_std[self.n_a * s + a]
            #print(std_prior, r_sigma)
            denom = 1./(std_prior * std_prior) + 1./(r_sigma * r_sigma) if r_sigma!=0 else 0
            self.r_mean[s *self.n_a+ a ] = ((1./(std_prior * std_prior) * self.r_mean[s *self.n_a+ a ]  + r/(r_sigma * r_sigma))/denom) if denom!=0 else r
            self.r_std[s * self.n_a + a] = (np.sqrt(1./ denom)) if denom!=0 else 0
            #print(s, a, self.r_mean, self.r_std)
            #exit()
            self.s_0 = s_1
        #print(self.pprior)
        #print(self.r_mean)
        #print(self.r_std)






#python PSRL.py --rep 100 --episode 100  --numdata 1000 --rstd 1.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs = "?", type = int, default= 100, help = 'number of repetitions'  )
    parser.add_argument('--episode', nargs = "?", type = int, default= 100, help = 'number of episode'  )
    #parser.add_argument('--r0', nargs = "?", type = float, default = 1.0, help = 'value of r0'  )
    parser.add_argument('--numdata', nargs = "?", type = int, default= 1000, help = 'number of data'  )
    parser.add_argument('--rightprop', nargs = "?", type = float, default= 0.6, help = 'warm start random exploration right probability'  )
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')

    args = parser.parse_args()

    num_iter, gamma, n_s, n_a, num_rep = 200, 0.95, 5, 2, args.rep
    episodes = args.episode
    Total_data = args.numdata
    right_prop = args.rightprop
    #print(num_rep, episodes, Total_data, right_prop)

    r = np.zeros(n_s * n_a)
    r_vals = range(1,4)
    #r_vals = [5./1000]
    r_right = 10.0

    for r0_val in r_vals:
        r[0] = float(r0_val)
        r[-1] = r_right
        r_std = args.rstd
        print("reward standard deviation is {}".format(r_std))
        # r[0] = 10.
        # r[-1] = 0.1
        Q_0 = np.zeros(n_s * n_a)
        V_0 = np.zeros(n_s)
        rou = np.ones(n_s) / n_s
        p = np.zeros(n_s * n_a * n_s)
        print("r[0] and r[-1] are {}, {}".format(r[0], r[-1]))
        #exit()

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
        print("Q real is {}".format(Q_real))
        s_0 = 2

        Q_approximation = None
        initial_s_dist = "even"
        if initial_s_dist == "even":
            R_real = np.mean(V_real)
            initial_w = np.ones(n_s) / n_s
        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        numerical_tol = 1e-6
        S_0 = None

        ## PSRL data parameter specification
        print("total num of data is {}".format(Total_data))
        numdata_1 = Total_data * 3/10
        seq_if = False
        numdata_2 = Total_data - numdata_1

        print("# of epsisodes is {}".format(episodes))
        num_datas = [numdata_2 / episodes] * episodes
        #print(num_datas)
        CS_num = 0.
        future_V = np.zeros(num_rep)

        for i in range(num_rep):
            para_cl = parameter_prior(n_s,n_a, s_0)
            all_data = []
            if not seq_if:
                while True:
                    data1 = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop,  std = r_std)
                    p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data1, n_s, n_a)
                    Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                    V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
                    #print("first stage visiting frequency is {}".format(f_n))
                    if f_n.all()!=0:
                        break
            else:
                data1 = collect_data_swimmer.collect_data(p, r, numdata_2 / episodes, s_0, n_s, n_a, right_prop=right_prop,
                                                          std=r_std)
            #data =  collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop)
            all_data += data1
            para_cl.update(data1, r_sigma= r_std)
            Q_estimate = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
            #print(Q_estimate)
            second_stage_data = []
            for num_data in num_datas:
                data = collect_data_swimmer.collect_data(p, r, num_data, para_cl.s_0, n_s, n_a, Q= Q_estimate, epsilon= 0,  std = r_std)
                all_data+=data
                second_stage_data += data
                para_cl.update(data, r_sigma= r_std)
                Q_estimate = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
                #print(para_cl.pprior)
                #print(para_cl.r_mean)
            #exit()
            #print(Q_estimate)
            #print(para_cl.pprior)
            #print(para_cl.r_mean)
            #transition = np.array([1.] * n_s * (n_s * n_a))
            #for i in range(n_s):
            #    for j in range(n_a):
            #        transition[
            #        (i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)] = para_cl.pprior[(i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)] \
            #                                                                      / np.sum(para_cl.pprior[(i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)])
            #r_n = para_cl.r_mean
            #print(r_n)
            #print(transition)
            #Q_estimate = Iterative_Cal_Q.cal_Q_val(transition, Q_0, r_n, num_iter , gamma, n_s, n_a)
            #print(len(all_data))
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(all_data, n_s, n_a)
            Q_estimate = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            V_here = optimize_pfs.policy_val_iteration(Q_estimate, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
            FS_bool = optimize_pfs.FS_bool(Q_estimate, V_max_index, n_s, n_a)
            CS_num += FS_bool

            # 5.3
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, Total_data, s_0, num_iter, gamma,
                                                                 Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                 data = all_data)
            # print("{} th replication : Q_n is {}, CI len is {}".format(i, Q_n, CI_len))
            cov_bool_Q = np.logical_and(Q_real <= (Q_n + CI_len_Q + numerical_tol),
                                        Q_real >= (Q_n - CI_len_Q - numerical_tol))
            cov_bool_V = np.logical_and(V_real <= (V_n + CI_len_V + numerical_tol),
                                        V_real >= (V_n - CI_len_V - numerical_tol))
            cov_bool_R = np.logical_and(R_real <= (R_n + CI_len_R + numerical_tol),
                                        R_real >= (R_n - CI_len_R - numerical_tol))
            # print(cov_bool_Q)
            # exit()
            # print(cov_bool)
            cov_bools_Q += cov_bool_Q
            cov_bools_V += cov_bool_V
            cov_bools_R += cov_bool_R
            CI_lens_Q.append(CI_len_Q)
            CI_lens_V.append(CI_len_V)
            CI_lens_R.append(CI_len_R)


        PCS = np.float(CS_num) / num_rep
        CI_len = 1.96 * np.sqrt(PCS * (1 - PCS) / num_rep)
        fv = np.mean(future_V)
        fv_std = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        # print(CS_num_naive)
        print("PCS is {}, with CI length {}".format(PCS, CI_len))
        print("future value func is {} with  CI length {}, real value is {}, diff is {}".format(fv, 1.96*fv_std/ np.sqrt(num_rep), rv, diff))



        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.divide(cov_bools_Q, num_rep)
        cov_rate_V = np.divide(cov_bools_V, num_rep)
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)
        print("coverage for Q")
        print(cov_rate_Q)
        print(cov_rate_CI_Q)
        print("mean coverage for Q ")
        print(np.mean(cov_rate_Q))
        print(np.mean(cov_rate_CI_Q))
        print("coverage for V")
        print(cov_rate_V)
        print(cov_rate_CI_V)
        print("mean coverage for V")
        print(np.mean(cov_rate_V))
        print(np.mean(cov_rate_CI_V))
        print("coverage for R")
        print(cov_rate_R)
        print(cov_rate_CI_R)
        print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
        print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
        print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))






if __name__=="__main__":
    main()