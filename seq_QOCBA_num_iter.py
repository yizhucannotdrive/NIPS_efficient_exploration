import numpy as np
import Iterative_Cal_Q
import inference
import collect_data_swimmer
import optimize_pfs
import two_stage_inference
from scipy.optimize import minimize
import argparse
import time
import cal_impirical_r_p
from optimize_pfs import policy_val_iteration
from seq_Q_OCBA import parameter_prior
import matplotlib.pyplot as plt


#python seq_QOCBA_num_iter.py --rep 1000 --r0 2.0 --optLb 1e-6 --numdata 1000 --r_prior 1.0 --rightprop 0.6 --rstd 0.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    parser.add_argument('--r_prior', nargs="?", type=float, default=0.0, help='prior value of reward function')
    parser.add_argument('--optLb', nargs="?", type=float, default=1e-2, help='value of r0')
    parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    #parser.add_argument('--epi_step_num', nargs="?", type=int, default=100, help='number of episode steps')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')
    parser.add_argument('--opt_ori', nargs="?", type=bool, default=False,
                        help='Q-OCBA optimization method')
    parser.add_argument('--num_value_iter', nargs="?", type=int, default=200, help='number of value iteration')
    parser.add_argument('--opt_one_step', nargs="?", type=bool, default=False,
                        help='Q-OCBA optimization running only one step')

    args = parser.parse_args()
    opt_ori = args.opt_ori
    print("Q-OCBA optimization method using original formulation is {}".format(opt_ori))
    num_rep = args.rep
    optLb = args.optLb
    s_0 = 2
    # collect data configuration
    n_s = 5
    print("n_s is {}".format(n_s))
    n_a = 2
    # value-iteration configuration
    num_iter = 200
    gamma = 0.95
    # real p and r
    p = np.zeros(n_s * n_a * n_s)
    Q_0 = np.zeros(n_s * n_a)
    V_0 = np.zeros(n_s)
    rou = np.ones(n_s) / n_s
    r = np.zeros(n_s * n_a)
    r[0] = args.r0
    r[-1] = 10.
    r_sd = args.rstd
    r_prior_mean = args.r_prior
    print("reward standard deviation is {}".format(r_sd))
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
    right_prop = args.rightprop


    Total_data = args.numdata
    print("total num of data is {}".format(Total_data))
    numdata_1 = 5
    if r_sd == 0:
        episodes_num = [2, 5, 10, 15, 20]
        episodes_num = list(range(1, 15))
    else:
        episodes_num = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        episodes_num = list(range(1, 15))
        #episodes = [1]
    epi_num_results = []
    print("Seq_Q_OCBA")
    for epi_num in episodes_num:
        print("epsisode num is {}".format(epi_num))
        num_datas = [Total_data/ epi_num] * (epi_num)
        #num_datas = [1000, 0]
        CS_num = 0.
        future_V = np.zeros(num_rep)
        Total_time = []
        #if use Bayesian prior as exploration
        Bayes_resample = False
        #optLbs = np.linspace(optLb, 1e-6, len(num_datas))
        ##print(optLbs)
        #exit()
        for ii in range(num_rep):
            time_rep = time.time()
            para_cl = parameter_prior(n_s,n_a, s_0, r_mean_prior =  r_prior_mean)
            data =  collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop,  std = r_sd)
            para_cl.update(data, resample = Bayes_resample)
            p_n, r_n, r_std = para_cl.get_para( resample = Bayes_resample)
            var_r_n = r_std **2
            #print(p_n)
            #print(r_n)
            #print(r_std)

            #test
            #p_n = p
            #r_n = r

            Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, args.num_value_iter , gamma, n_s, n_a)
            V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
            for jj, num_data in enumerate(num_datas):
                TM = inference.embedd_MC(p_n, n_s, n_a, V_n_max_index)
                I = np.identity(n_s * n_a)
                I_TM = np.linalg.inv(I - gamma * TM)
                V = np.diag(var_r_n)
                ds = []
                ds_V = []
                for i in range(n_s):
                    for j in range(n_a):
                        p_sa = p_n[(i * n_a * n_s + j * n_s): (i * n_a * n_s + (j + 1) * n_s)]
                        dij = inference.cal_cov_p_quad_V(p_sa, V_n, n_s)
                        ds.append(dij)
                        if j == V_n_max_index[i]:
                            ds_V.append(dij)
                D = np.diag(ds)
                cov_V_D = V + D
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
                #bb = np.asarray(b)


                if opt_ori:
                    def fun(x):
                        return -x[0]
                else:
                    def fun(x):
                        return x[0]
                constraints = []
                if opt_ori:
                    for i in range(n_s):
                        for j in range(n_a):
                            if j != V_n_max_index[i]:
                                # print(denom_consts[i][j])
                                if np.max(denom_consts[i][j]) > 1e-5:
                                    constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: up_c / (
                                        np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) - x[0],
                                                        'args': (quad_consts[i][j], denom_consts[i][j])})
                else:
                    for i in range(n_s):
                        for j in range(n_a):
                            if j != V_n_max_index[i]:
                                # print(denom_consts[i][j])
                                if np.max(quad_consts[i][j]) > 1e-5:
                                    constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: -(
                                        np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) / up_c + x[0],
                                                        'args': (quad_consts[i][j], denom_consts[i][j])})

                for i in range(AA.shape[0]):
                    constraints.append(
                        {'type': 'eq', 'fun': lambda x, a, b: np.dot(a, x[1:]) - b, 'args': (AA[i], b[i])})
                constraints = tuple(constraints)
                bnds = []
                bnds.append((0., None))
                for i in range(n_s * n_a):
                    bnds.append((optLb, 1))
                    #bnds.append((optLbs[jj], 1))

                bnds = tuple(bnds)
                initial = np.ones(n_s * n_a + 1) / (n_s * n_a)

                initial[0] = 0.1
                # print(initial)
                # print("number of equality constraints is {}".format(len(A)))
                if args.opt_one_step:
                    res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                                   constraints=constraints, options = {'disp':False, 'maxiter':1})
                else:
                    res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                                   constraints=constraints)
                x_opt = res.x[1:]

                #exit()

                #print("***", para_cl.s)


                data = collect_data_swimmer.collect_data(p, r, num_data, para_cl.s, n_s, n_a, pi_s_a=x_opt,  std = r_sd)
                para_cl.update(data, resample = Bayes_resample)
                _, _, freq, _ = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
                #print("x_opt", x_opt)
                #print("freq", freq)
                #dist = np.linalg.norm(freq - x_opt)
                #dist = sklearn.metrics.mutual_info_score(freq, x_opt)
                #print(dist)

                p_n, r_n, r_std = para_cl.get_para(resample = Bayes_resample)
                var_r_n = r_std ** 2
                Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, args.num_value_iter, gamma, n_s, n_a)
                V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
            #print(p_n, r_n)
            #print(Q_n)
            Total_time.append(time.time() - time_rep)
            V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
            future_V[ii] = np.dot(rou, V_here)
            fS_bool = optimize_pfs.FS_bool(Q_n, V_max_index, n_s, n_a)
            CS_num += fS_bool
            fv = np.mean(future_V)
            fv_std = np.std(future_V)
            rv = np.dot(rou, V_real)
            diff = rv - fv
        PCS = np.float(CS_num) / num_rep
        CI_len = 1.96 * np.sqrt(PCS * (1 - PCS) / num_rep)
        print("PCS is {}, with CI length {}".format(PCS, CI_len))
        print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv, 1.96 * fv_std / np.sqrt(
            num_rep), rv, diff))
        runnung_time_mean = np.mean(Total_time)
        runnung_time_CI  = 1.96 * np.std(Total_time)/ np.sqrt(num_rep)
        #3print("average running time of Seq QOCBA is {} with CI length {}".format(runnung_time_mean, runnung_time_CI))
        #exit()
        epi_num_results.append(PCS)


    # follow original
    """
    CS_num_naive = 0
    future_V = np.zeros(num_rep)
    for i in range(num_rep):
        data = collect_data_swimmer.collect_data(p, r, Total_data, s_0, n_s, n_a, right_prop=0.6, std = r_sd)
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        # print(Q_n)
        V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
        # print(V_here, V_real)
        future_V[i] = np.dot(rou, V_here)
        fS_bool_ = optimize_pfs.FS_bool(Q_n, V_max_index, n_s, n_a)
        CS_num_naive += fS_bool_
        # if not FS_bool_:
        # print(i)
        # print(f_n)
        # print(Q_n)
    PCS_naive = np.float(CS_num_naive) / num_rep
    CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
    fv = np.mean(future_V)
    fv_std = np.std(future_V)
    rv = np.dot(rou, V_real)
    diff = rv - fv
    print("follow original")
    print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
    print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv, 1.96 * fv_std / np.sqrt(
        num_rep), rv, diff))
    """
    plt.plot(episodes_num, epi_num_results, 'b>--', markersize=6, label="Q-OCBA")
    # plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2), color='b', alpha=0.4)
    # plt.axhline(y=0.95)
    plt.xlabel("Number of Iterations")
    # plt.ylabel("CR overall coverage")
    plt.ylabel("PCS")
    plt.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.title(r'$\sigma_R= {}, r_L = {}, n = {},$ PCS'.format(r_sd, r[0], Total_data))
    #plt.axhline(y=PCS_naive)
    plt.show()






if __name__=="__main__":
    main()