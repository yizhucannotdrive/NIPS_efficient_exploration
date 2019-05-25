import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
import cvxpy as cp
import two_stage_inference
from scipy.optimize import minimize
import compare_var


def FS_bool(Q_n, V_max_index, n_s, n_a):
    for i in range(n_s):
       if V_max_index[i] !=np.argmax(Q_n[i*n_a: (i+1) * n_a]):
           return False
    return True


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
def main():

    two_stage_opt_bool  = False
    print("two_stage_opt_bool is {}".format(two_stage_opt_bool))
    two_stage_eps_greedy_bool = True
    print("two_stage_eps_greedy_bool is {}".format(two_stage_eps_greedy_bool))
    num_rep = 1000
    initial_s_dist = "even"
    Q_approximation = None
    right_prop = 0.8
    s_0 = 2
    # collect data configuration
    num_data =  10000
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
    r = np.zeros(n_s * n_a)
    r[0] = 3.
    r[-1] = 10.
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
    print("Q real is {}".format(Q_real))
    if initial_s_dist == "even":
        R_real = np.mean(V_real)
    initial_w = np.ones(n_s) / n_s
    if two_stage_opt_bool or two_stage_eps_greedy_bool:
        while True:
            p_n, r_n, f_n, var_r_n, Q_n, V_n, V_n_max_index, R_n = two_stage_inference.stage_1_estimation(p, r, num_data_1, s_0, n_s, n_a, Q_0,
                                                                                  right_prop, num_iter, gamma, initial_w)
            if f_n.all()!=0:
                break
        print(Q_n)
        I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = inference.get_Sigma_n_comp(p_n, f_n, var_r_n, V_n, gamma,
                                                                                              n_s, n_a, V_n_max_index)
        if two_stage_opt_bool:
            quad_consts = np.zeros((n_s, n_a))
            denom_consts = np.zeros((n_s, n_a, n_s * n_a))

            for i in range(n_s):
                for j in range(n_a):
                    if j!= V_n_max_index[i]:
                        minus_op = np.zeros(n_s * n_a)
                        minus_op[i*n_a + j] =1
                        minus_op[i * n_a + V_n_max_index[i]] =  -1
                        denom_consts[i][j] = np.power(np.dot(minus_op, I_TM), 2) * np.diag(cov_V_D)
                        quad_consts[i][j] = (Q_n[i*n_a + j] - Q_n[i*n_a + V_n_max_index[i]])**2

            A, b, G, h = two_stage_inference.construct_contrain_matrix(p_n, n_s, n_a)
            AA = np.array(A)
            bb = np.asarray(b)
            def fun(x):
                return x[0]

            def cons(x, i,j):
                z = x[0]
                w = x[1:]
                return  quad_consts[i][j] / (np.sum(np.multiply(denom_consts[i][j], np.reciprocal(w)))) -z

            def eqcons(x,a, b):
                return np.dot(a,x[1:]) -b

            constraints = []
            for i in range(n_s):
                for j in range(n_a):
                    if j != V_n_max_index[i]:
                        constraints.append({'type': 'ineq', 'fun': lambda x , up_c, denom_c: up_c  / (np.sum(np.multiply( denom_c, np.reciprocal(x[1:])))) +x[0], 'args': (quad_consts[i][j], denom_consts[i][j])})

            for i in range(AA.shape[0]):
                constraints.append({'type': 'eq', 'fun': lambda x , a, b : np.dot(a,x[1:]) -b, 'args': (AA[i], b[i])})
            constraints = tuple(constraints)
            bnds = []
            for i in range(n_s * n_a +1):
                bnds.append((1e-10, None))
            bnds =  tuple(bnds)
            initial = np.ones(n_s * n_a + 1)/ (n_s * n_a)
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
                return z
            initial[0] = 0
            #print(initial)
            res = minimize(fun, initial,  method='SLSQP', bounds=bnds,
            constraints = constraints)
            x_opt=  res.x[1:]
            print(x_opt)
            #print(-res.x[0])

            opt_val = func_val(x_opt)
            print(opt_val)
            #print(f_n)
            epsilon = 0.3
            tran_M = transition_mat_S_A_epsilon(p_n, epsilon, V_n_max_index, n_s, n_a)
            bench_w = compare_var.solveStationary(tran_M)
            bench_w = np.array(bench_w).reshape(-1, )
            print(bench_w)
            bench_val =  func_val(bench_w)
            bench_val =  func_val(f_n)
            print(bench_val)
            #exit()



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
    epsilons = []
    for epsilon in epsilons:
        print("epsilon is {}".format(epsilon))
        CS_num_naive = 0
        for i in range(num_rep):
            data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,  Q = Q_n, epsilon= epsilon, print_pro_right = False)
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            Q_here = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
            #print(Q_here)
            FS_bool_ =  FS_bool(Q_here, V_max_index, n_s, n_a)
            CS_num_naive += FS_bool_
            #if not FS_bool_:
                #print(i)
                #print(f_n)
                #print(Q_here)
                #exit()
        PCS_naive = np.float(CS_num_naive)/num_rep
        CI_len = 1.96* np.sqrt(PCS_naive * (1-PCS_naive)/ num_rep)
        print(CS_num_naive)
        print(PCS_naive,CI_len)

    CS_num_naive = 0
    for i in range(num_rep):
        if two_stage_opt_bool:
            data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                     pi_s_a=x_opt)
        else:
            data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop=right_prop)
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        # print(Q_n)
        FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
        CS_num_naive += FS_bool_
        #if not FS_bool_:
            #print(i)
            #print(f_n)
           # print(Q_n)
    PCS_naive = np.float(CS_num_naive) / num_rep
    CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
    print(CS_num_naive)
    print(PCS_naive, CI_len)
    exit()

    two_stage_opt_bool = False
    CS_num_naive = 0
    print("follow original")
    for i in range(num_rep):
        if two_stage_opt_bool:
            data = collect_data_swimmer.collect_data(p, r, num_data_2, s_0, n_s, n_a, right_prop=right_prop,
                                                     pi_s_a=x_opt)
        else:
            data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop=right_prop)
        p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        # print(Q_n)
        FS_bool_ = FS_bool(Q_n, V_max_index, n_s, n_a)
        CS_num_naive += FS_bool_
        # if not FS_bool_:
        # print(i)
        # print(f_n)
        # print(Q_n)
    PCS_naive = np.float(CS_num_naive) / num_rep
    CI_len = 1.96 * np.sqrt(PCS_naive * (1 - PCS_naive) / num_rep)
    print(CS_num_naive)
    print(PCS_naive, CI_len)




if __name__ == "__main__":
    main()