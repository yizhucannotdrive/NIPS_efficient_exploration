import inference
import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import collect_data_swimmer
import cvxpy as cp
import two_stage_inference
from scipy.optimize import minimize
import compare_var

def main():
    initial_s_dist = "even"
    Q_approximation = None
    right_props = [0.4, 0.6, 0.7 ,0.8, 0.99]
    # collect data configuration
    n_s = 10
    print("n_s is {}".format(n_s))
    n_a = 2
    # value-iteration configuration
    num_iter = 200
    gamma = 0.95
    # real p and r
    p = np.zeros(n_s * n_a * n_s)
    Q_0 = np.zeros(n_s * n_a)
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
    Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    V_real, V_max_index = inference.get_V_from_Q(Q_real, n_s, n_a)
    print("Q real is {}".format(Q_real))
    if initial_s_dist == "even":
        R_real = np.mean(V_real)
    quad_consts = np.zeros((n_s, n_a))
    denom_consts = np.zeros((n_s, n_a, n_s * n_a))
    f_n = np.ones(n_s * n_a)
    var_r_n = np.zeros(n_s * n_a)
    I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = inference.get_Sigma_n_comp(p, f_n, var_r_n, V_real, gamma,
                                                                                          n_s, n_a, V_max_index)
    for i in range(n_s):
        for j in range(n_a):
            if j != V_max_index[i]:
                minus_op = np.zeros(n_s * n_a)
                minus_op[i * n_a + j] = 1
                minus_op[i * n_a + V_max_index[i]] = -1
                denom_consts[i][j] = np.power(np.dot(minus_op, I_TM), 2) * np.diag(cov_V_D)
                quad_consts[i][j] = (Q_real[i * n_a + j] - Q_real[i * n_a + V_max_index[i]]) ** 2

    A, b, G, h = two_stage_inference.construct_contrain_matrix(p, n_s, n_a)
    AA = np.array(A)

    def fun(x):
        return x[0]

    constraints = []
    for i in range(n_s):
        for j in range(n_a):
            if j != V_max_index[i]:
                constraints.append({'type': 'ineq', 'fun': lambda x, up_c, denom_c: up_c / (
                    np.sum(np.multiply(denom_c, np.reciprocal(x[1:])))) - x[0],
                                    'args': (quad_consts[i][j], denom_consts[i][j])})

    for i in range(AA.shape[0]):
        constraints.append({'type': 'eq', 'fun': lambda x, a, b: np.dot(a, x[1:]) - b, 'args': (AA[i], b[i])})
    constraints = tuple(constraints)
    bnds = []
    for i in range(n_s * n_a + 1):
        bnds.append((0.000001, None))
    bnds = tuple(bnds)
    initial = np.ones(n_s * n_a + 1) / (n_s * n_a)
    initial[0] = 1
    # print(initial)
    res = minimize(fun, initial, method='SLSQP', bounds=bnds,
                   constraints=constraints)
    x_opt = res.x[1:]
    print(x_opt)

    def func_val(x):
        vals = []
        for i in range(n_s):
            for j in range(n_a):
                if j != V_max_index[i]:
                    vals.append(quad_consts[i][j] / (2 *
                        np.sum(np.multiply(denom_consts[i][j], np.reciprocal(x)))))
        z = np.min(vals)
        # print (z)
        # print (vals)
        return z

    opt_val = func_val(x_opt)
    print(opt_val)
    n= np.array([10000, 50000, 100000])
    print(1 - np.exp(-n * opt_val))
    for right_prop in right_props:
        transition_p = compare_var.transition_mat_S_A (p, right_prop, n_s, n_a)
        f_n = compare_var.solveStationary( transition_p )
        print(right_prop)
        #print(f_n)
        bench_val = func_val(f_n)
        print(bench_val)
        print(1- np.exp(-n * bench_val))
    # exit()





if __name__== "__main__":
    main()