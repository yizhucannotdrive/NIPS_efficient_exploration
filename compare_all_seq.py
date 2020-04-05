import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
# import cvxpy as cp
import two_stage_inference
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


# python compare_all_seq.py --rep 100 --r0 2.0 --optLb 1e-6 --rstd 0.0 --iflog 0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    parser.add_argument('--optLb', nargs="?", type=float, default=1e-2, help='value of r0')
    # parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')
    parser.add_argument('--opt_ori', nargs="?", type=int, default=0,
                        help='Q-OCBA optimization method')
    parser.add_argument('--episode', nargs="?", type=int, default=100, help='number of episode')
    parser.add_argument('--epi_step_num', nargs="?", type=int, default=100, help='number of episode steps')
    parser.add_argument('--first_stage_data', nargs="?", type=int, default=3, help='number of first stage data')
    parser.add_argument('--r_prior', nargs="?", type=float, default=0.0, help='prior value of reward function')
    parser.add_argument('--opt_one_step', nargs="?", type=int, default=0,
                        help='Q-OCBA optimization running only one step')
    parser.add_argument('--iflog', nargs="?", type=int, default=0,
                        help='whether take logrithm of x-axis')

    args = parser.parse_args()




    opt_ori = True if args.opt_ori else False
    print("Q-OCBA optimization method using original formulation is {}".format(opt_ori))
    num_rep = args.rep
    right_prop = args.rightprop
    print("right prop is {}".format(right_prop))

    optLb = args.optLb
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
            num_datas = list(range(0, 8000, 1000))
        else:
            num_datas = list(range(0, 4000, 500))
            #num_datas = list(range(0, 2000, 500))
    else:
        num_datas = [0, 100, 1000, 5000, 10000]
    #num_datas = list(range(1000, 5000, 2000))
    #num_datas = [2000]
    QOCBAs_PCS = []
    REs_PCS = []
    QOCBAs_fr = []
    REs_fr = []
    eps_PCSs = []
    eps_frs = []
    UCRL_PCSs = []
    UCRL_frs = []
    PSRL_PCSs = []
    PSRL_frs = []
    REs_PCS_08= []
    REs_fr_08 = []


    Bayes_resample = False
    print_if = True
    epsilon = 0.2

    print("epsilon is {}".format(epsilon))

    for num_data in num_datas:
        print("numdata is {}".format(num_data))
        stage_datas = [episode_steps] * (num_data / episode_steps)
        CS_num_naive = 0
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            para_cl = seq_cls(n_s, n_a, s_0, r_mean_prior=r_prior_mean)
            data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            para_cl.update(data, resample=False)
            p_n, r_n, r_std = para_cl.get_para(resample=False)
            Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            for num_dat in stage_datas:
                stage_data = collect_data_swimmer.collect_data(p, r, num_dat, s_0, n_s, n_a, right_prop=right_prop, Q=Q_here,
                                                         epsilon=epsilon, print_pro_right=False, std=r_sd)
                para_cl.update(stage_data, resample=Bayes_resample)
                p_n, r_n, r_std = para_cl.get_para(resample=False)
                Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            # print(Q_here)
            V_here = policy_val_iteration(Q_here, n_s, n_a, V_0, num_iter, r, p, gamma)
            future_V[i] = np.dot(rou, V_here)
            FS_bool_ = FS_bool(Q_here, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_

        PCS_naive = np.float(CS_num_naive) / num_rep
        CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
        fv = np.mean(future_V)
        fv_std = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        eps_PCSs.append(PCS_naive)
        eps_frs.append(diff)
        if print_if:
            print("epsilon--greedy with epsilon {}:".format(epsilon))
            print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
            print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                   1.96 * fv_std / np.sqrt(
                                                                                                       num_rep), rv,
                                                                                                   diff))
        # exit()

        CS_num_naive = 0
        future_V = np.zeros(num_rep)
        for iii in range(num_rep):
            para_cl = seq_cls(n_s, n_a, s_0, r_mean_prior=r_prior_mean)
            data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            #data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=0.3, std=r_sd)
            para_cl.update(data, resample=Bayes_resample)
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
                # bb = np.asarray(b)

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
                    # bnds.append((optLbs[jj], 1))

                bnds = tuple(bnds)
                initial = np.ones(n_s * n_a + 1) / (n_s * n_a)

                initial[0] = 0.1
                # print(initial)
                # print("number of equality constraints is {}".format(len(A)))
                if args.opt_one_step:
                    #print("haha")
                    res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                                   constraints=constraints, options={'disp': False, 'maxiter': 1})
                else:
                    #print("huha")
                    res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                                   constraints=constraints)
                x_opt = res.x[1:]

                # exit()

                # print("***", para_cl.s)
                #print(x_opt)

                data = collect_data_swimmer.collect_data(p, r, stage_data, para_cl.s, n_s, n_a, pi_s_a=x_opt, std=r_sd)
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
            V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
            future_V[iii] = np.dot(rou, V_here)
            FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_
        PCS_naive = np.float(CS_num_naive) / num_rep
        CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
        fv = np.mean(future_V)
        fv_std = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        # print(CS_num_naive)
        QOCBAs_PCS.append(PCS_naive)
        QOCBAs_fr.append(diff)
        if print_if:
            print("Q-OCBA:")
            print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
            print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                   1.96 * fv_std / np.sqrt(
                                                                                                       num_rep), rv, diff))

        # follow original
        CS_num_naive = 0
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            data = collect_data_swimmer.collect_data(p, r, num_data + numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            # print(Q_n)
            V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
            FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_
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
        REs_PCS.append(PCS_naive)
        REs_fr.append(diff)
        if print_if:
            print("follow original")
            print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
            print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                   1.96 * fv_std / np.sqrt(
                                                                                                    num_rep), rv, diff))

        # follow original
        CS_num_naive = 0
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            data = collect_data_swimmer.collect_data(p, r, num_data + numdata_1, s_0, n_s, n_a,
                                                     right_prop=0.8, std=r_sd)
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            # print(Q_n)
            V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
            FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_
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
        REs_PCS_08.append(PCS_naive)
        REs_fr_08.append(diff)
        if print_if:
            print("RE(0.8)")
            print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
            print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                   1.96 * fv_std / np.sqrt(
                                                                                                       num_rep), rv,
                                                                                                   diff))






        #UCRL
        #delta = 0.05
        CS_num = 0.
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            data = collect_data_swimmer.collect_data(p, r, numdata_1, s_0, n_s, n_a, right_prop=right_prop, std=r_sd)
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
            datahere = data + UCRL_cl.datas
            #print(UCRL_cl.t, num_data)
            #print(len(datahere))
            #exit()
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(datahere, n_s, n_a)
            Q_estimate = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            # print(Q_estimate)
            FS_bool_ = FS_bool(Q_estimate, V_max_index, n_s, n_a)
            CS_num += FS_bool_
            V_here = policy_val_iteration(Q_estimate, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
        PCS = np.float(CS_num) / num_rep
        CI_len = 1.96 * np.sqrt(PCS * (1 - PCS) / num_rep)
        fv = np.mean(future_V)
        fv_std = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        # print(CS_num_naive)
        UCRL_PCSs.append(PCS)
        UCRL_frs.append(diff)
        if print_if:
            print("UCRL")
            print("PCS is {}, with CI length {}".format(PCS, CI_len))
            print("future value func is {} with  CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                    1.96 * fv_std / np.sqrt(
                                                                                                        num_rep), rv, diff))


        CS_num = 0.
        future_V = np.zeros(num_rep)
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
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            Q_estimate = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            V_here = policy_val_iteration(Q_estimate, n_s, n_a, V_0, num_iter, r, p, gamma)
            # print(V_here, V_real)
            future_V[i] = np.dot(rou, V_here)
            FS_bool_ = FS_bool(Q_estimate, V_max_index, n_s, n_a)
            CS_num += FS_bool_
        PCS = np.float(CS_num) / num_rep
        CI_len = 1.96 * np.sqrt(PCS * (1 - PCS) / num_rep)
        fv = np.mean(future_V)
        fv_std = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        PSRL_PCSs.append(PCS)
        PSRL_frs.append(diff)
        # print(CS_num_naive)
        if print_if:
            print("PSRL")
            print("PCS is {}, with CI length {}".format(PCS, CI_len))
            print("future value func is {} with  CI length {}, real value is {}, diff is {}".format(fv,
                                                                                                    1.96 * fv_std / np.sqrt(
                                                                                                        num_rep), rv,
                                                                                                    diff))
    print("epsilon greedy")
    print(eps_PCSs)
    print(eps_frs)
    print("QOCBA")
    print(QOCBAs_PCS)
    print(QOCBAs_fr)
    print("REs")
    print(REs_PCS)
    print(REs_fr)
    print("RE(0.8)s")
    print(REs_PCS_08)
    print(REs_fr_08)
    print("UCRL ")
    print(UCRL_PCSs)
    print(UCRL_frs)
    print("PSRL ")
    print(PSRL_PCSs)
    print(PSRL_frs)
    if logif:
        num_datas = np.log(np.array(num_datas)+1)
    plt.plot(num_datas, eps_PCSs, 'k<--', markersize=6, label="epsilon-greedy")
    plt.plot(num_datas, UCRL_PCSs, 'm+--', markersize=6, label="UCRL")
    plt.plot(num_datas, PSRL_PCSs, 'cx--', markersize=6, label="PSRL")
    plt.plot(num_datas, QOCBAs_PCS, 'ro--', markersize=6, label="Q-OCBA")
    # plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1), color='r', alpha=0.4)
    plt.plot(num_datas, REs_PCS, 'b>--', markersize=6, label="RE({})".format(right_prop))
    plt.plot(num_datas, REs_PCS_08, 'g*--', markersize=6, label="RE(0.8)")

    # plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2), color='b', alpha=0.4)
    # plt.axhline(y=0.95)
    plt.xlabel("total number of data")
    # plt.ylabel("CR overall coverage")
    plt.ylabel("PCS")
    plt.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.title(r'$\sigma_R= {}, r_L = {}$ PCS'.format(r_sd, r[0]))
    plt.show()

    plt.plot(num_datas, eps_frs, 'k<--', markersize=6, label="epsilon-greedy")
    plt.plot(num_datas, UCRL_frs, 'm+--', markersize=6, label="UCRL")
    plt.plot(num_datas, PSRL_frs, 'cx--', markersize=6, label="PSRL")
    plt.plot(num_datas, REs_fr_08, 'g*--', markersize=6, label="RE(0.8)")


    plt.plot(num_datas, QOCBAs_fr, 'ro--', markersize=6, label="Q-OCBA")
    # plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1), color='r', alpha=0.4)
    plt.plot(num_datas, REs_fr, 'b>--', markersize=6, label="RE({})".format(right_prop))
    # plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2), color='b', alpha=0.4)
    # plt.axhline(y=0.95)
    plt.xlabel("total number of data")
    # plt.ylabel("CR overall coverage")
    plt.ylabel("future regret")
    plt.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.title(r'$\sigma_R= {}, r_L = {}$ future regret'.format(r_sd, r[0]))

    plt.show()


if __name__ == "__main__":
    main()