import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
#import cvxpy as cp
import two_stage_inference
from scipy.optimize import minimize
import compare_var
import time
import argparse
import functools

def FS_bool(Q_n, V_max_index, n_s, n_a):
    for i in range(n_s):
       if V_max_index[i] !=np.argmax(Q_n[i*n_a: (i+1) * n_a]):
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


def transition_mat_S_A_epsilon (p, epsilon, V_max_index, n_s, n_a):
    tran_M = np.zeros((n_s * n_a ,n_s * n_a))
    for i in range(n_s):
        for j in range(n_a):
            for k in range(n_s):
                for l in range(n_a):
                    #tran_M[i*n_a + j , k*n_a + l] = stat_p[i] * ((2*right_prop-1)*j + 1-right_prop) * p[i*n_a* n_s + j * n_s + k] * ((2*right_prop-1)*l + 1-right_prop)
                    if l== V_max_index[k]:
                        right_prop_2 = 1-epsilon
                    else:
                        right_prop_2 = epsilon/(n_a-1)
                    tran_M[i*n_a + j , k*n_a + l] =  p[i*n_a* n_s + j * n_s + k]  * right_prop_2
    return tran_M
#python optimize_pfs.py --rep 100 --r0 2.0 --optLb 0.05 --numdata 10000 --rstd 1.0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs="?", type=int, default=100, help='number of repetitions')
    parser.add_argument('--r0', nargs="?", type=float, default=1.0, help='value of r0')
    parser.add_argument('--optLb', nargs="?", type=float, default=1e-2, help='value of r0')
    parser.add_argument('--numdata', nargs="?", type=int, default=1000, help='number of data')
    parser.add_argument('--rightprop', nargs="?", type=float, default=0.6,
                        help='warm start random exploration right probability')
    parser.add_argument('--rstd', nargs="?", type=float, default=1.0,
                        help='standard deviation of reward')
    parser.add_argument('--opt_ori', nargs="?", type=bool, default=False,
                        help='Q-OCBA optimization method')
    args = parser.parse_args()
    opt_ori = args.opt_ori
    print("Q-OCBA optimization method using original formulation is {}".format(opt_ori))

    two_stage_opt_bool  = True
    print("two_stage_opt_bool is {}".format(two_stage_opt_bool))
    two_stage_eps_greedy_bool = True
    print("two_stage_eps_greedy_bool is {}".format(two_stage_eps_greedy_bool))
    num_rep = args.rep
    initial_s_dist = "even"
    Q_approximation = None
    right_prop = args.rightprop
    optLb = args.optLb
    s_0 = 2
    # collect data configuration
    num_data =  args.numdata
    num_data_1 = num_data * 3/10
    num_data_2 = num_data * 7/10
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
    V_0 = np.zeros(n_s)
    rou = np.ones(n_s) / n_s
    r = np.zeros(n_s * n_a)
    r[0] = args.r0
    r[-1] = 10.
    r_std =args.rstd
    print("reward standard deviation is {}".format(r_std))
    #r[0] = 10.
    #r[-1] = 0.1
    print("r[0] and r[-1] are {}, {}".format(r[0], r[-1] ))
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

    if two_stage_opt_bool or two_stage_eps_greedy_bool:
        Q_ns = []
        x_opts = []
        counts = []
        data1s = []
        PCS_first_stage = 0.
        times_run = np.zeros(4)
        for i in range(num_rep):
            count = 0
            while True:
                count+=1
                data1 = collect_data_swimmer.collect_data(p, r, num_data_1, s_0, n_s, n_a, right_prop=right_prop, std = r_std)
                p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data1, n_s, n_a)
                t = time.time()
                Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
                t1 = time.time() -t
                V_n, V_n_max_index = inference.get_V_from_Q(Q_n, n_s, n_a)
                #print("first stage visiting frequency is {}".format(f_n))
                if f_n.all()!=0:
                    break
            counts.append(count)
            data1s.append(data1)
            PCS_first_stage += functools.reduce(lambda i, j: i and j,  map(lambda i,j: i==j, V_max_index, V_n_max_index), True)
            Q_ns.append(Q_n)
            #print("first stage trial = {}".format(count))
            #print("real V_max_index vs estimated V_max_index after first stage is {} and {}".format(V_max_index, V_n_max_index))
            #print(Q_n)
            #test
            #p_n = p
            #V_n = V_real
            #V_n_max_index = V_max_index
            t = time.time()
            I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = inference.get_Sigma_n_comp(p_n, f_n, var_r_n, V_n, gamma, n_s, n_a, V_n_max_index)
            t2 = time.time()-t
            # test  covariance
            #cov_V_D = np.diag(np.ones(n_s * n_a))
            #print("first stage stationary dist is {}".format(f_n))
            #print("real Q is {}".format(Q_real))
            #print("Q_n estiamte is {}".format(Q_n))
            #Q_n = Q_real

            if two_stage_opt_bool:
                quad_consts = np.zeros((n_s, n_a))
                denom_consts = np.zeros((n_s, n_a, n_s * n_a))
                t = time.time()
                for i in range(n_s):
                    for j in range(n_a):
                        if j!= V_n_max_index[i]:
                            minus_op = np.zeros(n_s * n_a)
                            minus_op[i*n_a + j] =1
                            minus_op[i * n_a + V_n_max_index[i]] =  -1
                            c1 = np.power(np.dot(minus_op, I_TM), 2)
                            denom_consts[i][j] = c1 * np.diag(cov_V_D)
                            #print(I_TM, c1)
                            #exit()
                            quad_consts[i][j] = (Q_n[i*n_a + j] - Q_n[i*n_a + V_n_max_index[i]])**2


                A, b, G, h = two_stage_inference.construct_contrain_matrix(p_n, n_s, n_a)
                AA = np.array(A)
                t3 = time.time()-t
                #bb = np.asarray(b)
                if opt_ori:
                    def fun(x):
                        return -x[0]
                else:
                    def fun(x):
                        return x[0]
                """
                def cons(x, i,j):
                    z = x[0]
                    w = x[1:]
                    return  quad_consts[i][j] / (np.sum(np.multiply(denom_consts[i][j], np.reciprocal(w)))) -z
    
                def eqcons(x,a, b):
                    return np.dot(a,x[1:]) -b
                """
                #print("quardratic coeff of opt is {}".format(quad_consts))
                #print("denom consts coef of opt is {}".format(denom_consts))

                constraints = []
                if opt_ori:
                    for i in range(n_s):
                        for j in range(n_a):
                            if j != V_n_max_index[i]:
                                #print(denom_consts[i][j])
                                if np.max(denom_consts[i][j]) > 1e-5:
                                    constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: up_c / (
                                        np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) - x[0],
                                                        'args': (quad_consts[i][j], denom_consts[i][j])})
                else:
                    for i in range(n_s):
                        for j in range(n_a):
                            if j != V_n_max_index[i]:
                                #print(denom_consts[i][j])
                                if np.max(denom_consts[i][j]) > 1e-5:
                                    constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: -(np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) / up_c + x[0],
                                                        'args': (quad_consts[i][j], denom_consts[i][j])})





                for i in range(AA.shape[0]):
                    constraints.append({'type': 'eq', 'fun': lambda x , a, b : np.dot(a,x[1:]) -b, 'args': (AA[i], b[i])})
                constraints = tuple(constraints)
                bnds = []
                bnds.append((0., None))
                for i in range(n_s * n_a ):
                    bnds.append((optLb, 1))
                bnds =  tuple(bnds)
                initial = np.ones(n_s * n_a + 1)/ (n_s * n_a)

                initial[0] = 0.1
                #print(initial)
                t_1 = time.time()
                #print("number of equality constraints is {}".format(len(A)))
                t = time.time()
                res = minimize(fun, initial,  method='SLSQP', bounds=bnds,
                constraints = constraints)
                x_opt=  res.x[1:]
                t4 = time.time() - t
                runnung_t = time.time()-t_1

                def func_val(x):
                    vals = []
                    for i in range(n_s):
                        for j in range(n_a):
                            if j != V_n_max_index[i]:
                                vals.append(quad_consts[i][j] / ( 2 *
                                    np.sum(np.multiply(denom_consts[i][j], np.reciprocal(x)))))
                    z = np.min(vals)
                    #print (z)
                    #print (vals)
                    #z = 1
                    return z
                #print("optimization running time is {}".format(runnung_t))


                #ec = np.dot(AA, x_opt) - b
                #print("last equality constraint coeff is {}, {}".format(AA[-1], b[-1]))
                #print("verify equality constraints, equality residual is {}".format(ec))

                #opt_val = func_val(x_opt)
                #print(f_n)

                epsilon = 0.3
                tran_M = transition_mat_S_A_epsilon(p_n, epsilon, V_n_max_index, n_s, n_a)
                bench_w = compare_var.solveStationary(tran_M)
                bench_w = np.array(bench_w).reshape(-1, )
                times_run+= np.array([t1, t2, t3, t4])
                #print(bench_w)
                #bench_val_1=  func_val(bench_w)
                #bench_val_2 =  func_val(f_n)
                #print("optimal exploration policy has stationary dist {} with sum {}".format(x_opt, np.sum(x_opt)))
                #print("optimal value is {}".format(res.x[0]))
                #print("optimal value with optimal solution is {} ".format(opt_val))
                #print("benchmark objective value is {} and {}".format(bench_val_1, bench_val_2))
                #exit()
            x_opts.append(x_opt)
        mean_count = np.mean(counts)
        mean_running_time =  times_run/num_rep
        print("running time of value iteration, parameter calculation, optimization parameter calculation, optimization solving is {}".format(mean_running_time))
        std_count = np.std(counts)
        print("first stage average # of trials is {} with CI length  {}".format(mean_count, 1.96 * std_count/np.sqrt(num_rep)))
        PFS_first_stage = 1- PCS_first_stage/num_rep
        print("PFS after first stage is {} ".format(PFS_first_stage))


    """
    w = cp.Variable(n_s * n_a)
    #z = cp.Variable(1)
    rate = w[0*n_a + 0]
    for i in range(n_s):
        for j in range(n_a):
            if j!= V_n_max_index[i]:
                #rates.append(quad_consts[i][j] * cp.inv_pos(cp.sum(cp.multiply(denom_consts[i][j], cp.inv_pos(w)))))
                rate = cp.min(rate, w[i*n_a + j])
    #rates = np.array(rates)
    problem = cp.Problem(cp.Maximize(rate), [AA * w == bb, w >= 0])
    problem.solve()
    # Print result.
    print("\nThe optimal value is", problem.value)
    print("A solution w is")
    print(w.value)
    exit()
    """


    epsilons  = [0.3, 0.2, 0.1,0.01]
    epsilons = [0.2]
    epsilons = []
    for epsilon in epsilons:
        print("epsilon is {}".format(epsilon))
        CS_num_naive = 0
        future_V = np.zeros(num_rep)
        for i in range(num_rep):
            Q_n = Q_ns[i]
            data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,  Q = Q_n, epsilon= epsilon, print_pro_right = False)
            data = data + data1s[i]
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            #print(Q_here)
            V_here = policy_val_iteration(Q_here, n_s, n_a, V_0, num_iter, r, p, gamma)
            future_V[i] = np.dot(rou, V_here)
            FS_bool_ =  FS_bool(Q_here, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_
            #if not FS_bool_:
                #print(i)
                #print(f_n)
                #print(Q_here)
                #exit()
        PCS_naive = np.float(CS_num_naive)/num_rep
        CI_len = 1.96* np.sqrt(PCS_naive * (1-PCS_naive)/ num_rep)
        fv = np.mean(future_V)
        fv_std  = np.std(future_V)
        rv = np.dot(rou, V_real)
        diff = rv - fv
        print("epsilon--greedy with epsilon {}:".format(epsilon))
        print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
        print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv, 1.96*fv_std/ np.sqrt(num_rep), rv, diff))

    #exit()

    CS_num_naive = 0
    future_V = np.zeros(num_rep)
    for i in range(num_rep):
        x_opt = x_opts[i]
        #print(x_opt)
        if two_stage_opt_bool:
            second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                     pi_s_a=x_opt)
        else:
            second_data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop)

        data = second_data + data1s[i]
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        V_here = policy_val_iteration(Q_n, n_s, n_a, V_0, num_iter, r, p, gamma)
        #print(V_here, V_real)
        future_V[i] = np.dot(rou, V_here)
        FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
        CS_num_naive += FS_bool_

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
        #if not FS_bool_:
            #print(i)
            #print(f_n)
           # print(Q_n)
    PCS_naive = np.float(CS_num_naive) / num_rep
    CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
    fv = np.mean(future_V)
    fv_std = np.std(future_V)
    rv = np.dot(rou, V_real)
    diff = rv - fv
    #print(CS_num_naive)
    print("Q-OCBA:")
    print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
    print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv, 1.96*fv_std/ np.sqrt(num_rep), rv, diff))

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

    exit()


    # follow original
    CS_num_naive = 0
    future_V = np.zeros(num_rep)
    for i in range(num_rep):
        data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop)
        data = data+ data1s[i]
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
    print("follow original")
    print("PCS is {}, with CI length {}".format(PCS_naive, CI_len))
    print("future value func is {} with CI length {}, real value is {}, diff is {}".format(fv,1.96*fv_std/ np.sqrt(num_rep),  rv, diff))




if __name__ == "__main__":
    main()