import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
# import cvxpy as cp
import two_stage_inference
from cvxopt import matrix, log, div, spdiag, solvers
from scipy.optimize import minimize
import compare_var
import time
import argparse
import functools
import matplotlib.pyplot as plt
from PSRL import parameter_prior as PSRLcls
import UCRL_2
from seq_Q_OCBA import parameter_prior as seq_cls

def FS_bool(Q_n, V_max_index, n_s, n_a):
    for i in range(n_s):
        if V_max_index[i] != np.argmax(Q_n[i * n_a: (i + 1) * n_a]):
            return False
    return True


def policy_val_iteration(Q_n, n_s, n_a, V_init, num_iter, r, p, gamma):
    _, V_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
    V_current = np.copy(V_init)
    V_pre = np.copy(V_init)
    for t in range(num_iter):
        for i in range(n_s):
            exp_next_q = 0
            for k in range(n_s):
                exp_next_q += p[i * n_s * n_a + V_max_index[i] * n_s + k] * V_pre[k]
            V_current[i] = r[i * n_a + V_max_index[i]] + gamma * exp_next_q
        # print("estimate of {} th iteration is {}".format(t,Q_current))
        V_pre = np.copy(V_current)
    return V_current


def transition_mat_S_A_epsilon(p, epsilon, V_max_index, n_s, n_a):
    tran_M = np.zeros((n_s * n_a, n_s * n_a))
    for i in range(n_s):
        for j in range(n_a):
            for k in range(n_s):
                for l in range(n_a):
                    # tran_M[i*n_a + j , k*n_a + l] = stat_p[i] * ((2*right_prop-1)*j + 1-right_prop) * p[i*n_a* n_s + j * n_s + k] * ((2*right_prop-1)*l + 1-right_prop)
                    if l == V_max_index[k]:
                        right_prop_2 = 1 - epsilon
                    else:
                        right_prop_2 = epsilon / (n_a - 1)
                    tran_M[i * n_a + j, k * n_a + l] = p[i * n_a * n_s + j * n_s + k] * right_prop_2
    return tran_M


# python compare_all_coverage_seq.py --rep 100 --r0 2.0 --rstd 0.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    # parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')
    parser.add_argument('--episode', nargs="?", type=int, default=100, help='number of episode')
    parser.add_argument('--epi_step_num', nargs="?", type=int, default=100, help='number of episode steps')
    parser.add_argument('--first_stage_data', nargs="?", type=int, default=100, help='number of first stage data')
    parser.add_argument('--r_prior', nargs="?", type=float, default=0.0, help='prior value of reward function')
    parser.add_argument('--iflog', nargs="?", type=int, default=0,
                        help='whether take logrithm of x-axis')

    args = parser.parse_args()




    num_rep = args.rep
    right_prop = args.rightprop
    print("right prop is {}".format(right_prop))
    s_0 = 2
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
    R_real = np.mean(V_real)
    # print("Q real is {}".format(Q_real))
    #num_datas = list(range(500, 10500, 500))
    episode_steps = args.epi_step_num

    #numdata_1 = episode_steps
    numdata_1 = args.first_stage_data
    print("first stage data num is {}".format(numdata_1))
    print("epsisode timestep is {}".format(episode_steps))
    logif = True if  args.iflog else False
    print("we print x axis in log is {}".format(logif))
    if not logif:
        if r_sd ==10.0:
            num_datas = list(range(10, 8000, 1000))
        else:
            num_datas = list(range(5, 10010, 1000))
    else:
        num_datas = [10, 100, 1000, 5000, 10000]
    #num_datas = list(range(1000, 5000, 2000))
    #num_datas = [2000]
    QOCBAs_Q_cov = []
    REs_Q_cov = []
    eps_Q_cov = []
    UCRL_Q_cov = []
    PSRL_Q_cov = []
    Bayes_resample = False
    print_if = True
    epsilon = 0.2
    S_0 = None
    initial_w = np.ones(n_s) / n_s
    numerical_tol = 1e-6
    Q_approximation = None

    print("epsilon is {}".format(epsilon))

    for num_data in num_datas:
        print("numdata is {}".format(num_data))
        stage_datas = [episode_steps] * (num_data / episode_steps)

        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        print("epsilon greedy")
        for i in range(num_rep):
            para_cl = seq_cls(n_s, n_a, s_0, r_mean_prior=r_prior_mean)
            all_data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            para_cl.update(all_data, resample=False)
            p_n, r_n, r_std = para_cl.get_para(resample=False)
            Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            for num_dat in stage_datas:
                stage_data = collect_data_swimmer.collect_data(p, r, num_dat, s_0, n_s, n_a, right_prop=right_prop, Q=Q_here,
                                                         epsilon=epsilon, print_pro_right=False, std=r_sd)
                para_cl.update(stage_data, resample=Bayes_resample)
                p_n, r_n, r_std = para_cl.get_para(resample=False)
                Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                all_data += stage_data
            # print(Q_here)
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter,
                                                                           gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=all_data)
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


        #print(CI_lens_Q)
        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.mean(np.divide(cov_bools_Q, num_rep))
        cov_rate_V = np.mean(np.divide(cov_bools_V, num_rep))
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)

        eps_Q_cov.append(cov_rate_Q)

        if print_if:
            print("mean coverage for Q ")
            print(np.mean(cov_rate_Q))
            print(np.mean(cov_rate_CI_Q))
            print("mean coverage for V")
            print(np.mean(cov_rate_V))
            print(np.mean(cov_rate_CI_V))
            print("coverage for R")
            print(cov_rate_R)
            print(cov_rate_CI_R)
            print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
            print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
            print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))
        # exit()
        print("Q-OCBA")
        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        for iii in range(num_rep):
            para_cl = seq_cls(n_s, n_a, s_0, r_mean_prior=r_prior_mean)
            all_data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            #data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=0.3, std=r_sd)
            para_cl.update(all_data, resample=Bayes_resample)
            p_n, r_n, r_std = para_cl.get_para(resample=Bayes_resample)
            var_r_n = r_std ** 2
            # print(p_n)
            # print(r_n)
            # print(r_std)

            # test
            # p_n = p
            # r_n = r

            Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
            for jj, stage_data in enumerate(stage_datas):
                TM_V = inference.P_V(p_n, n_s, n_a, V_n_max_index)
                I_V = np.identity(n_s)
                I_TM_V = np.linalg.inv(I_V - gamma * TM_V)


                var_r_n_V = np.array([var_r_n[i * n_a + V_n_max_index[i]] for i in range(n_s)])
                V_V = np.diag(var_r_n_V)

                ds = []
                ds_V = []
                for i in range(n_s):
                    for j in range(n_a):
                        p_sa = p_n[(i * n_a * n_s + j * n_s): (i * n_a * n_s + (j + 1) * n_s)]
                        dij = inference.cal_cov_p_quad_V(p_sa, V_n, n_s)
                        ds.append(dij)
                        if j == V_n_max_index[i]:
                            ds_V.append(dij)
                D_V = np.diag(ds_V)
                cov_V_V_D = V_V + D_V





                A, b, G, h = two_stage_inference.construct_contrain_matrix(p_n, n_s, n_a)
                AA = np.array(A)
                # bb = np.asarray(b)

                quad_con_vec = np.power(np.dot(initial_w, I_TM_V), 2) * np.diag(cov_V_V_D)
                # print(quad_con_vec)
                # if  not np.all(f_n):
                #    print(f_n)
                #    print(quad_con_vec)
                #    print("need more data for first stage")
                #    exit()
                quad_con_vec_all = np.zeros(n_s * n_a)
                for i in range(n_s):
                    quad_con_vec_all[i * n_a + V_n_max_index[i]] = quad_con_vec[i]
                # print(quad_con_vec)

                # print(I_TM_V)
                # print(cov_V_V_D)
                # print(initial_w)
                # Create a new model
                init_v_opt = 1. / (n_a * n_s)
                quad_con_vec_all = matrix(quad_con_vec_all)
                array_quad_con_vec = np.array(quad_con_vec_all).transpose()[0]

                # print(array_quad_con_vec)
                # exit()
                def F(x):
                    u = np.divide(1, x)
                    # print(u)
                    uu = np.multiply(array_quad_con_vec, u)
                    # print(quad_con_vec_all)
                    # print(uu)
                    val = np.sum(uu)
                    # print(val)
                    return val

                A, b, G, h = two_stage_inference.construct_contrain_matrix(p_n, n_s, n_a)
                AA = np.array(A)
                bb = np.asarray(b)

                constraints = []

                for i in range(AA.shape[0]):
                    constraints.append({'type': 'eq', 'fun': lambda x, a, b: np.dot(a, x) - b, 'args': (AA[i], bb[i])})
                constraints = tuple(constraints)
                bnds = []
                for i in range(n_s * n_a):
                    bnds.append((1e-6, None))
                    # bnds.append((0.001, None))

                bnds = tuple(bnds)
                initial = np.ones(n_s * n_a) / (n_s * n_a)
                # print(initial)
                res = minimize(F, initial, method='SLSQP', bounds=bnds,
                               constraints=constraints)
                x_opt = res.x

                # exit()

                # print("***", para_cl.s)
                #print(x_opt)

                data = collect_data_swimmer.collect_data(p, r, stage_data, para_cl.s, n_s, n_a, pi_s_a=x_opt, std=r_sd)
                all_data += data
                para_cl.update(data, resample=Bayes_resample)
                _, _, freq, _ = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
                # print("x_opt", x_opt)
                # print("freq", freq)
                # dist = np.linalg.norm(freq - x_opt)
                # dist = sklearn.metrics.mutual_info_score(freq, x_opt)
                # print(dist)

                p_n, r_n, r_std = para_cl.get_para(resample=Bayes_resample)
                var_r_n = r_std ** 2
                Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter,
                                                                           gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=all_data)
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

        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.mean(np.divide(cov_bools_Q, num_rep))
        cov_rate_V = np.mean(np.divide(cov_bools_V, num_rep))
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)

        QOCBAs_Q_cov.append(cov_rate_Q)

        if print_if:
            print("mean coverage for Q ")
            print(np.mean(cov_rate_Q))
            print(np.mean(cov_rate_CI_Q))
            print("mean coverage for V")
            print(np.mean(cov_rate_V))
            print(np.mean(cov_rate_CI_V))
            print("coverage for R")
            print(cov_rate_R)
            print(cov_rate_CI_R)
            print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
            print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
            print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))

        # follow original
        print("random exploration")
        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        for i in range(num_rep):
            data = collect_data_swimmer.collect_data(p, r, num_data + numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter,
                                                                           gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=data)
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

        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.mean(np.divide(cov_bools_Q, num_rep))
        cov_rate_V = np.mean(np.divide(cov_bools_V, num_rep))
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)

        REs_Q_cov.append(cov_rate_Q)

        if print_if:
            print("mean coverage for Q ")
            print(np.mean(cov_rate_Q))
            print(np.mean(cov_rate_CI_Q))
            print("mean coverage for V")
            print(np.mean(cov_rate_V))
            print(np.mean(cov_rate_CI_V))
            print("coverage for R")
            print(cov_rate_R)
            print(cov_rate_CI_R)
            print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
            print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
            print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))
        #UCRL
        #delta = 0.05

        print("UCRL")
        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        for i in range(num_rep):
            all_data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            pre_collected_stats = UCRL_2.get_pre_collected_stats(data, n_s, n_a)
            UCRL_cl = UCRL_2.UCRL(n_s, n_a, 0.05, numdata_1, s_0, num_data, pre_collected_stats)
            while UCRL_cl.t < num_data:
                UCRL_cl.update_point_estimate_and_CIbound()
                # print("step1 finished")
                UCRL_cl.Extended_Value_Iter()
                # print("step2 finished")
                UCRL_cl.collect_data_and_update(p, r, r_std=r_sd)
                # print("step3 finished")
                # print(UCRL_cl.t)
            UCRL_cl.update_point_estimate_and_CIbound()
            all_data = all_data + UCRL_cl.datas
            #print(UCRL_cl.t, num_data)
            #print(len(datahere))
            #exit()
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter,
                                                                           gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=all_data)
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

        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.mean(np.divide(cov_bools_Q, num_rep))
        cov_rate_V = np.mean(np.divide(cov_bools_V, num_rep))
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)

        UCRL_Q_cov.append(cov_rate_Q)

        if print_if:
            print("mean coverage for Q ")
            print(np.mean(cov_rate_Q))
            print(np.mean(cov_rate_CI_Q))
            print("mean coverage for V")
            print(np.mean(cov_rate_V))
            print(np.mean(cov_rate_CI_V))
            print("coverage for R")
            print(cov_rate_R)
            print(cov_rate_CI_R)
            print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
            print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
            print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))

        print("PSRL")
        cov_bools_Q = np.zeros(n_s * n_a)
        cov_bools_V = np.zeros(n_s)
        cov_bools_R = 0.
        # print("Q real is {}".format(Q_real))
        # print("V real is {}".format(V_real))
        # print("R real is {}".format(R_real))
        CI_lens_Q = []
        CI_lens_V = []
        CI_lens_R = []
        for i in range(num_rep):
            para_cl = PSRLcls(n_s, n_a, s_0)
            data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            para_cl.update(data, r_sigma=r_sd)
            Q_estimate = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
            # print(Q_estimate)
            for nd in stage_datas:
                dat = collect_data_swimmer.collect_data(p, r, nd, para_cl.s_0, n_s, n_a, Q=Q_estimate,
                                                         epsilon=0, std=r_sd)
                data += dat
                para_cl.update(dat, r_sigma=r_sd)
                Q_estimate = para_cl.sampled_MDP_Q(Q_0, num_iter, gamma)
                # print(para_cl.pprior)
                # print(para_cl.r_mean)
            # exit()
            # print(Q_estimate)
            # print(para_cl.pprior)
            # print(para_cl.r_mean)
            # transition = np.array([1.] * n_s * (n_s * n_a))
            # for i in range(n_s):
            #    for j in range(n_a):
            #        transition[
            #        (i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)] = para_cl.pprior[(i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)] \
            #                                                                      / np.sum(para_cl.pprior[(i * n_s * n_a + j * n_s): (i * n_s * n_a + (j + 1) * n_s)])
            # r_n = para_cl.r_mean
            # print(r_n)
            # print(transition)
            # Q_estimate = Iterative_Cal_Q.cal_Q_val(transition, Q_0, r_n, num_iter , gamma, n_s, n_a)
            Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0,
                                                                           num_iter,
                                                                           gamma,
                                                                           Q_0, n_s, n_a, r, p, initial_w, right_prop,
                                                                           data=data)
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

        CI_len_Q_mean = np.mean(CI_lens_Q)
        CI_len_V_mean = np.mean(CI_lens_V)
        CI_len_R_mean = np.mean(CI_lens_R)
        CI_len_Q_ci = 1.96 * np.std(CI_lens_Q) / np.sqrt(num_rep)
        CI_len_V_ci = 1.96 * np.std(CI_lens_V) / np.sqrt(num_rep)
        CI_len_R_ci = 1.96 * np.std(CI_lens_R) / np.sqrt(num_rep)

        cov_rate_Q = np.mean(np.divide(cov_bools_Q, num_rep))
        cov_rate_V = np.mean(np.divide(cov_bools_V, num_rep))
        cov_rate_R = np.divide(cov_bools_R, num_rep)
        cov_rate_CI_Q = 1.96 * np.sqrt(cov_rate_Q * (1 - cov_rate_Q) / num_rep)
        cov_rate_CI_V = 1.96 * np.sqrt(cov_rate_V * (1 - cov_rate_V) / num_rep)
        cov_rate_CI_R = 1.96 * np.sqrt(cov_rate_R * (1 - cov_rate_R) / num_rep)

        PSRL_Q_cov.append(cov_rate_Q)

        if print_if:
            print("mean coverage for Q ")
            print(np.mean(cov_rate_Q))
            print(np.mean(cov_rate_CI_Q))
            print("mean coverage for V")
            print(np.mean(cov_rate_V))
            print(np.mean(cov_rate_CI_V))
            print("coverage for R")
            print(cov_rate_R)
            print(cov_rate_CI_R)
            print("CI len for Q CI {} with ci {}".format(CI_len_Q_mean, CI_len_Q_ci))
            print("CI len for V CI {} with ci {}".format(CI_len_V_mean, CI_len_V_ci))
            print("CI len for R CI {} with ci {}".format(CI_len_R_mean, CI_len_R_ci))

    print("epsilon greedy")
    print(eps_Q_cov)

    print("QOCBA")
    print(QOCBAs_Q_cov)

    print("REs")
    print(REs_Q_cov)

    print("UCRL ")
    print(UCRL_Q_cov)

    print("PSRL ")
    print(PSRL_Q_cov)

    if logif:
        num_datas = np.log(np.array(num_datas)+1)
    plt.plot(num_datas, eps_Q_cov, 'g<--', markersize=6, label="epsilon-greedy")
    plt.plot(num_datas, UCRL_Q_cov, 'm+--', markersize=6, label="UCRL")
    plt.plot(num_datas, PSRL_Q_cov, 'cx--', markersize=6, label="PSRL")
    plt.plot(num_datas, QOCBAs_Q_cov, 'ro--', markersize=6, label="Q-OCBA")
    # plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1), color='r', alpha=0.4)
    plt.plot(num_datas, REs_Q_cov, 'b>--', markersize=6, label="RE({})".format(right_prop))
    # plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2), color='b', alpha=0.4)
    # plt.axhline(y=0.95)
    plt.xlabel("total number of data")
    # plt.ylabel("CR overall coverage")
    plt.ylabel("CI Coverage")
    plt.axhline(y=0.95)
    plt.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.title(r'$\sigma_R= {}, r_L = {}$ CI coverage'.format(r_sd, r[0]))
    plt.show()

if __name__ == "__main__":
    main()