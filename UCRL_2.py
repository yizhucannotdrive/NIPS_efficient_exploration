import numpy as np
import Iterative_Cal_Q
import inference
import collect_data_swimmer
import optimize_pfs
import cal_impirical_r_p
import argparse

class UCRL:

    def __init__ (self, n_s=0, n_a=0, delta = 0.05, t= 0, s_0 =2, Total = 10000, pre_colleted_stats = None):
        self.n_s = n_s
        self.n_a = n_a

        self.delta = delta
        self.rew = np.zeros(self.n_s * self.n_a)
        self.rew_CI_bound = np.zeros(self.n_s * self.n_a)
        self.transition = np.zeros(self.n_s * self.n_s * self.n_a)
        self.tran_CI_bound = np.zeros(self.n_s * self.n_a)
        self.t = t
        self.s = s_0
        self.policy = [0] * self.n_s
        self.Total = Total
        self.datas = []
        if pre_colleted_stats:
            self.count, self.srew_sa, self.trancount = pre_colleted_stats
        else:
            self.count = np.zeros(self.n_s * self.n_a)
            self.srew_sa = np.zeros(self.n_s * self.n_a)
            self.trancount = np.zeros(self.n_s * self.n_s * self.n_a)



    def update_point_estimate_and_CIbound(self):
        count= np.maximum(1, self.count)
        self.rew = self.srew_sa/ count
        self.transition = self.trancount/ np.repeat(count, self.n_s)
        self.rew_CI_bound = np.sqrt(7*np.log(2 * self.n_s * self.n_a * self. t / self.delta) / (2 * count))
        self.tran_CI_bound = np.sqrt(14 * self.n_s * np.log(2 * self.n_a * self. t / self.delta) /  count)

    def inner_max(self, s, a,  u):
        p_sa = self.transition[(s * self.n_s * self.n_a + a * self.n_s): (s * self.n_s * self.n_a + a * self.n_s + self.n_s) ]
        sort_index = np.argsort(u)[::-1]
        p_return = np.array([p for p in p_sa])
        p_return[sort_index[0]] = min(1, p_sa[sort_index[0]] + self.tran_CI_bound[s * self.n_a + a]/2)
        l = len(sort_index)-1
        while sum(p_return) >1:
            s_l = sort_index[l]
            p_return [ s_l ] = max(0, 1- (sum(p_return) - p_return[s_l]))
            l-=1
        return p_return


    def Extended_Value_Iter(self):
        val_u = np.zeros(self.n_s)
        while True:
            next_val_u = np.zeros(self.n_s)
            for s in range(self.n_s):
                u_s_tmp = -float('inf')
                for a in range(self.n_a):
                    tmp = self.rew[s * self.n_a + a] + self.rew_CI_bound[s * self.n_a + a] + np.dot(self.inner_max(s,a, val_u), val_u)
                    if tmp > u_s_tmp:
                        u_s_tmp, self.policy[s]= tmp, a
                next_val_u[s] = u_s_tmp
            #print(max(next_val_u - val_u) - min(next_val_u - val_u), 1 / np.sqrt(self.t))
            if max(next_val_u - val_u) - min(next_val_u - val_u) < 1/np.sqrt(self.t):
                self.val_u = next_val_u
                break
            else:
                val_u = np.array([u for u in  next_val_u])

    def collect_data_and_update(self, real_p, real_r, r_std =1):
        v = np.zeros(self.n_s * self.n_a)
        while v[self.s * self.n_a + self.policy[self.s]] < max(1, self.count[self.s * self.n_a + self.policy[self.s]]) :
            p_sa = real_p[(self.s * self.n_a * self.n_s +  self.policy[self.s] * self.n_s): (self.s * self.n_a * self.n_s + ( self.policy[self.s] + 1) * self.n_s)]
            s_new = int(np.random.choice(self.n_s, 1, p=p_sa)[0])
            r = np.random.normal(real_r[self.s * self.n_a + self.policy[self.s]], r_std)
            v[self.s * self.n_a + self.policy[self.s]]+=1
            self.t+=1
            data_i = (int(self.s), self.policy[self.s], r, s_new)
            self.datas.append(data_i)
            #print(data_i)
            self.s = s_new
            self.srew_sa[self.s * self.n_a + self.policy[self.s]] += r
            self.count[self.s * self.n_a + self.policy[self.s]] += 1
            self.trancount[self.s * self.n_a * self.n_s + self.policy[self.s] * self.n_s + s_new] += 1
            if self.t > self.Total:
                break




def get_pre_collected_stats(data, n_s, n_a):
    num_data = len(data)
    p_n = np.zeros(n_s * n_a * n_s)
    r_n = np.zeros(n_s * n_a)
    f_n = np.zeros(n_s * n_a)
    for i in range(num_data):
        data_i = data[i]
        s = int(data_i[0])
        a = int(data_i[1])
        r = data_i[2]
        s_next = int(data_i[3])
        p_n[s * n_s * n_a + a * n_s + s_next] += 1
        r_n[s * n_a + a] += r
        f_n[s * n_a + a] += 1
    return (f_n, r_n, p_n)

# python UCRL_2.py --rep 100  --numdata 1000 --rstd 1.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    #parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward ')

    args = parser.parse_args()
    num_iter, gamma, n_s, n_a, delta,  num_rep = 200, 0.95, 5, 2, 0.05, args.rep
    right_prop = args.rightprop
    Q_0 = np.zeros(n_s * n_a)
    V_0 = np.zeros(n_s)
    p = np.zeros(n_s * n_a * n_s)
    r = np.zeros(n_s * n_a)
    for r0_val in range(1, 4):
        r[0] = float(r0_val)
        r[-1] = 10.
        r_std  = args.rstd
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
        print("Q real is {}".format(Q_real))
        s_0 = 2
        rou = np.ones(n_s) / n_s

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

        ## UCRL
        CS_num = 0.
        num_data = args.numdata
        num_1  = num_data * 3/10
        num_2 = num_data * 7/10
        #print("smaller")
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            #all_data  = []
            while True:
                data1 = collect_data_swimmer.collect_data(p, r, num_1, s_0, n_s, n_a, right_prop=right_prop,  std = r_std)
                p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data1, n_s, n_a)
                Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
                #print("first stage visiting frequency is {}".format(f_n))
                if f_n.all()!=0:
                    break
            #all_data += data1
            pre_collected_stats = get_pre_collected_stats(data1, n_s, n_a)
            UCRL_cl = UCRL(n_s, n_a, 0.05, num_1, s_0, num_data, pre_collected_stats)
            while UCRL_cl.t < num_data:
                UCRL_cl.update_point_estimate_and_CIbound()
                #print("step1 finished")
                UCRL_cl.Extended_Value_Iter()
                #print("step2 finished")
                UCRL_cl.collect_data_and_update(p,r, r_std = r_std)
                #print("step3 finished")
                #print(UCRL_cl.t)
            UCRL_cl.update_point_estimate_and_CIbound()
            Q_estimate =  Iterative_Cal_Q.cal_Q_val(UCRL_cl.transition, Q_0, UCRL_cl.rew, num_iter , gamma, n_s, n_a)
            #print(Q_estimate)
            FS_bool = optimize_pfs.FS_bool(Q_estimate, V_max_index, n_s, n_a)
            CS_num += FS_bool
            V_here = optimize_pfs.policy_val_iteration(Q_estimate, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
            datahere = data1 + UCRL_cl.datas
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter, gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=datahere)
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
        print("future value func is {} with  CI length {}, real value is {}, diff is {}".format(fv, 1.96 * fv_std / np.sqrt(
            num_rep), rv, diff))

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

if __name__ == "__main__":
    main()

