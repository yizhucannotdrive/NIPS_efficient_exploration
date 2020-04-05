
import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
from cvxopt import matrix, log, div, spdiag, solvers
import sympy
import cvxpy as cp
from scipy.optimize import minimize
import compare_var
import optimize_pfs
import argparse


def stage_1_estimation(p, r, num_data_1, s_0, n_s, n_a, Q_0, right_prop, num_iter, gamma, initial_w):
    count = 0
    while True:
        count += 1
        data = collect_data_swimmer.collect_data(p, r, num_data_1, s_0, n_s, n_a, right_prop=right_prop)
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
        # print("first stage visiting frequency is {}".format(f_n))
        if f_n.all() != 0:
            break
    R_n = np.dot(initial_w, V_n)
    return p_n, r_n, f_n, var_r_n, Q_n, V_n, V_n_max_index, R_n


def stage_2_estimation(p, r, num_data_1, s_0, n_s, n_a, Q_0, x_opt, num_iter, gamma, initial_w):
    data = collect_data_swimmer.collect_data(p, r, num_data_1, s_0, n_s, n_a, pi_s_a=x_opt)
    p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
    Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
    V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
    R_n = np.dot(initial_w, V_n)
    return p_n, r_n, f_n, var_r_n, Q_n, V_n, V_n_max_index, R_n


def construct_contrain_matrix(p, n_s, n_a):
    A = []
    A.append(np.ones(n_s * n_a))
    b = np.zeros(n_s)
    b[0] = 1
    for i in range(n_s - 1):
        p_vec = p[range(i, n_s * n_a * n_s, n_s)]
        p_vec[i * n_a: (i + 1) * n_a] -= 1
        A.append(p_vec)
    A = np.array(A)
    # A = matrix(A)
    # b= matrix(b)
    G = matrix(-np.identity(n_s * n_a))
    h = matrix(np.zeros(n_s * n_a))
    return A, b, G, h


# python two_stage_inference.py --rep 1000 --r0 1.0  --numdata 10000
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    args = parser.parse_args()

    num_rep = args.rep
    initial_s_dist = "even"
    Q_approximation = None
    # Q_approximation = "linear_interpolation"
    right_prop = args.rightprop  # 0.8
    s_0 = 2
    # collect data configuration
    num_data = args.numdata
    num_data_1 = num_data * 3 / 10
    num_data_2 = num_data * 7 / 10
    print("num_data in stage 1 is {}, num_data in stage 2 is {}, rightprop in stage 1 is {}".format(num_data_1,
                                                                                                    num_data_2,
                                                                                                    right_prop))
    n_s = 5
    print("n_s is {}".format(n_s))
    n_a = 2
    # value-iteration configuration
    num_iter = 200
    gamma = 0.95
    # real p and r
    p = np.zeros(n_s * n_a * n_s)
    Q_0 = np.zeros(n_s * n_a)
    r = np.zeros(n_s * n_a)
    r[0] = args.r0
    r[-1] = 10.
    # r[0] = 10.
    # r[-1] = 0.1
    print(r)
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

    # one replication of coverage test
    # Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    # V_real = get_V_from_Q(Q_real, n_s, n_a)
    # Q_n, CI_len, V_n = get_CI(collec_data_bool, num_data, s_0, num_iter, gamma, Q_0, n_s, n_a, r, p)
    # print(Q_real)
    # print(V_real)
    # print(Q_n)
    # print(V_n)
    # print(CI_len)
    Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    V_real, V_max_index = inference.get_V_from_Q(Q_real, n_s, n_a)
    print("Q real is {}".format(Q_real))
    if initial_s_dist == "even":
        R_real = np.mean(V_real)
    initial_w = np.ones(n_s) / n_s
    opts = []
    datas = []
    Q_ns = []
    opts_ori = []
    for i in range(num_rep):
        while True:

            while True:
                data = collect_data_swimmer.collect_data(p, r, num_data_1, s_0, n_s, n_a, right_prop=right_prop)
                p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
                Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
                # print("first stage visiting frequency is {}".format(f_n))
                if f_n.all() != 0:
                    break
            datas.append(data)
            Q_ns.append(Q_n)
            I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = inference.get_Sigma_n_comp(p_n, f_n, var_r_n,
                                                                                                  V_n, gamma, n_s, n_a,
                                                                                                  V_n_max_index)

            # print( np.diag(cov_V_V_D))
            # exit()
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

            A, b, G, h = construct_contrain_matrix(p_n, n_s, n_a)
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
            # print(x_opt)

            opt_val = F(x_opt)

            # ori-Q-OCBA

            def fun(x):
                return x[0]

            # print("quardratic coeff of opt is {}".format(quad_consts))
            # print("denom consts coef of opt is {}".format(denom_consts))

            quad_consts = np.zeros((n_s, n_a))
            denom_consts = np.zeros((n_s, n_a, n_s * n_a))
            for i in range(n_s):
                for j in range(n_a):
                    if j != V_n_max_index[i]:
                        minus_op = np.zeros(n_s * n_a)
                        minus_op[i * n_a + j] = 1
                        minus_op[i * n_a + V_n_max_index[i]] = -1
                        c1 = np.power(np.dot(minus_op, I_TM), 2)
                        denom_consts[i][j] = c1 * np.diag(cov_V_D)
                        # print(I_TM, c1)
                        # exit()
                        quad_consts[i][j] = (Q_n[i * n_a + j] - Q_n[i * n_a + V_n_max_index[i]]) ** 2

            constraints = []
            for i in range(n_s):
                for j in range(n_a):
                    if j != V_n_max_index[i]:
                        # print(denom_consts[i][j])
                        if np.max(denom_consts[i][j]) > 1e-5:
                            constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: -(
                                np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) / up_c + x[0],
                                                'args': (quad_consts[i][j], denom_consts[i][j])})

            for i in range(AA.shape[0]):
                constraints.append({'type': 'eq', 'fun': lambda x, a, b: np.dot(a, x[1:]) - b, 'args': (AA[i], b[i])})
            constraints = tuple(constraints)
            bnds = []
            bnds.append((0., None))
            for i in range(n_s * n_a):
                bnds.append((1e-6, 1))
            bnds = tuple(bnds)
            initial = np.ones(n_s * n_a + 1) / (n_s * n_a)

            initial[0] = 0.1
            # print(initial)
            res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                           constraints=constraints)
            x_opt_ori = res.x[1:]
            opts_ori.append(x_opt_ori)
            bench_val = F(x_opt_ori)
            if bench_val > opt_val:
                break
            else:
                print(opt_val)
                print(bench_val)
                print("#####")

        # print(opt_val)
        # print(bench_val)
        # print("#####")

        epsilon = 0.3
        tran_M = optimize_pfs.transition_mat_S_A_epsilon(p_n, epsilon, V_n_max_index, n_s, n_a)
        bench_w = compare_var.solveStationary(tran_M)
        bench_w = np.array(bench_w).reshape(-1, )
        # print(bench_w)
        bench_val = F(bench_w)
        # print(bench_val)
        opts.append(x_opt)
        # exit()
    Q_approximation = None
    initial_s_dist = "even"
    if initial_s_dist == "even":
        R_real = np.mean(V_real)
        initial_w = np.ones(n_s) / n_s
        rou = initial_w
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

    for i in range(num_rep):
        x_opt = opts[i]
        second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                        pi_s_a=x_opt)
        data = second_data + datas[i]
        # data = second_data
        Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0, num_iter,
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

    # Q-OCBA ori

    CS_num_naive = 0
    cov_bools_Q = np.zeros(n_s * n_a)
    cov_bools_V = np.zeros(n_s)
    cov_bools_R = 0.
    # print("Q real is {}".format(Q_real))
    # print("V real is {}".format(V_real))
    # print("R real is {}".format(R_real))
    CI_lens_Q = []
    CI_lens_V = []
    CI_lens_R = []
    future_V = np.zeros(num_rep)
    for i in range(num_rep):
        x_opt = opts_ori[i]
        # print(x_opt)
        second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                        pi_s_a=x_opt)

        data = second_data + datas[i]
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0, num_iter,
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
    # print(CS_num_naive)
    print("Q-OCBA:")
    print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
    print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv, 1.96 * fv_std / np.sqrt(
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

    #exit()

    epsilons = [0.2]
    for epsilon in epsilons:
        print("epsilon is {}".format(epsilon))
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
            Q_n = Q_ns[i]
            second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                            Q=Q_n,
                                                            epsilon=epsilon, print_pro_right=False)
            data = second_data + datas[i]
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

    print("RE(0.8)")
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
        second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop)
        data = second_data + datas[i]
        Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R = inference.get_CI(Q_approximation, S_0, num_data, s_0, num_iter,
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