import numpy as np
import Iterative_Cal_Q
import cal_impirical_r_p
import pickle
import collect_data_swimmer
from scipy.stats import t
import math
numerical_tol = 1e-6

def find_arg_max_Q(Q_n, n_s, n_a):
    arg_ms = []
    for i in range(n_s):
        arg_m = np.argmax(Q_n[i*n_a: (i+1)*n_a])
        arg_ms.append(arg_m)
    return arg_ms




def embedd_MC( p_n,n_s, n_a, V_n_max_index):
    tot_states_num =  n_s * n_a
    TM = np.zeros((tot_states_num, tot_states_num))
    for i in range(n_s):
        for j in range(n_a):
            row_num = i * n_a + j
            for k in range(n_s):
                col_num = k * n_a + V_n_max_index[k]
                TM[row_num, col_num] = p_n[i * n_a * n_s + j * n_s + k]
    return TM
def P_V(p_n,n_s, n_a, V_n_max_index):
    M = np.zeros((n_s, n_s))
    for i in range(n_s):
        for j in range(n_s):
            M[i,j] = p_n[i * n_a * n_s + V_n_max_index[i] * n_s + j]
    return M

def cal_cov_p_quad_V(p_sa, V, n_s):
    cov_p = np.zeros((n_s, n_s))
    for i in range(n_s):
        for j in range(n_s):
            cov_p[i, j] = p_sa[i] * (1 - p_sa[i]) if i==j else -p_sa[i] * p_sa[j]
    v_p_sa = np.dot(np.dot(V, cov_p), V)
    return v_p_sa


def get_Sigma_n_comp(p_n, f_n, var_r_n, V_n, gamma, n_s, n_a,  V_n_max_index):
    TM = embedd_MC(p_n, n_s, n_a, V_n_max_index)
    I = np.identity(n_s * n_a)
    I_TM = np.linalg.inv(I - gamma * TM)

    TM_V = P_V(p_n, n_s, n_a, V_n_max_index)
    I_V = np.identity(n_s)
    I_TM_V = np.linalg.inv(I_V - gamma * TM_V)

    W_inverse = np.diag(1. / f_n)

    f_n_V = np.array([f_n[i * n_a + V_n_max_index[i]] for i in range(n_s)])
    W_inverse_V = np.diag(1. / f_n_V)

    V = np.diag(var_r_n)
    var_r_n_V = np.array([var_r_n[i * n_a + V_n_max_index[i]] for i in range(n_s)])
    V_V = np.diag(var_r_n_V)

    ds = []
    ds_V = []
    for i in range(n_s):
        for j in range(n_a):
            p_sa = p_n[(i * n_a * n_s + j * n_s): (i * n_a * n_s + (j + 1) * n_s)]
            dij = cal_cov_p_quad_V(p_sa, V_n, n_s)
            ds.append(dij)
            if j == V_n_max_index[i]:
                ds_V.append(dij)
    D = np.diag(ds)
    D_V = np.diag(ds_V)
    cov_V_D = V+D
    cov_V_V_D = V_V + D_V
    return I_TM, W_inverse, cov_V_D, I_TM_V,  W_inverse_V, cov_V_V_D

def cal_Sigma_n(p_n, f_n, var_r_n, V_n, gamma, n_s, n_a,  V_n_max_index, initial_w):
    I_TM, W_inverse, cov_V_D, I_TM_V, W_inverse_V, cov_V_V_D = get_Sigma_n_comp(p_n, f_n, var_r_n, V_n, gamma, n_s, n_a,  V_n_max_index)
    Sigma_n_Q = np.matrix(I_TM) * np.matrix(W_inverse * cov_V_D ) * np.matrix(I_TM).transpose()
    Sigma_n_V = np.matrix(I_TM_V) * np.matrix(W_inverse_V * (cov_V_V_D)) * np.matrix(I_TM_V).transpose()
    Sigma_n_R = np.dot(np.dot(initial_w, Sigma_n_V), initial_w)[0,0]
    #print(f_n)
    #print(Sigma_n_R)
    return Sigma_n_Q, Sigma_n_V, Sigma_n_R




def get_V_from_Q(Q, n_s, n_a):
    V_val = np.zeros(n_s)
    V_max_index = []
    for i in range(n_s):
        v_s = np.max(Q[i*n_a: (i+1) * n_a])
        max_index_s = np.argmax(Q[i*n_a: (i+1) * n_a])
        V_val[i] = v_s
        V_max_index.append(max_index_s)
    return V_val, V_max_index


def get_CI(Q_approximation, S_0, num_data, s_0, num_iter, gamma, Q_0,  n_s, n_a, r, p, initial_w, right_prop, pi_s_a = None, num_sec = 10, Q = None, epsilon = 0):

    if Q_approximation == None:
        # collect new data
        counter =0
        while True:
            counter+=1
            data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop=right_prop,  pi_s_a = pi_s_a, Q = Q, epsilon =epsilon)
            # get impirical statistics
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data, n_s, n_a)
            if counter > 3 or f_n.all()!=0 :
                #print("exploration fail: cannot visit every state action pair at least once")
                break
        # cal  Q_n  function and V_n function
        Q_n = Iterative_Cal_Q.cal_Q_val(p_n, Q_0, r_n, num_iter, gamma, n_s, n_a)
        V_n, V_n_max_index = get_V_from_Q(Q_n, n_s, n_a)
        R_n = np.dot(initial_w, V_n)
        Sigma_n_Q, Sigma_n_V, Sigma_n_R = cal_Sigma_n(p_n, f_n, var_r_n, V_n, gamma, n_s, n_a, V_n_max_index, initial_w)
        CI_len_Q = 1.96 * np.sqrt(np.diag(Sigma_n_Q)) / np.sqrt(num_data)
        CI_len_V = 1.96 * np.sqrt(np.diag(Sigma_n_V)) / np.sqrt(num_data)
        CI_len_R = 1.96 * np.sqrt(Sigma_n_R) / np.sqrt(num_data)
        return Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R

    if Q_approximation == "linear_interpolation":
        Q_ns = []
        R_ns = []
        V_ns = []
        # collect new data
        data = collect_data_swimmer.collect_data(p, r, num_data, s_0, n_s, n_a, right_prop=right_prop, pi_s_a = pi_s_a)
        sec_size = num_data / num_sec
        for i in range(num_sec):
            # get impirical statistics
            p_n, r_n, f_n, var_r_n = cal_impirical_r_p.cal_impirical_stats(data[i*sec_size: (i+1) * sec_size], n_s, n_a)
            #print(p_n, r_n)
            #print(f_n)
            Q_n = Iterative_Cal_Q.cal_Q_val_approx_linear_interpolation(p_n, Q_0, r_n, num_iter, gamma, S_0,  n_s, n_a)
            V_n, V_n_max_index = get_V_from_Q(Q_n, n_s, n_a)
            R_n = np.dot(initial_w, V_n)
            #print(Q_n)
            #print(R_n)
            Q_ns.append(Q_n )
            R_ns.append(R_n)
            V_ns.append(V_n)
        Q_bar = np.mean(Q_ns, 0)
        V_bar = np.mean(V_ns, 0)
        R_bar = np.mean(R_ns)
        Q_std = np.std(Q_ns, 0)
        V_std = np.std(V_ns, 0)
        R_std = np.std(R_ns)
        CI_len_Q = t.ppf(0.975, num_sec - 1) * Q_std/ np.sqrt(num_sec)
        CI_len_V = t.ppf(0.975, num_sec - 1) * V_std / np.sqrt(num_sec)
        CI_len_R = t.ppf(0.975, num_sec - 1) * R_std / np.sqrt(num_sec)
        #print(Q_bar)
        return Q_bar, CI_len_Q, V_bar, CI_len_V, R_bar, CI_len_R

def cal_coverage(Q_approximation,  num_data, s_0, num_iter, gamma, Q_0, n_s, n_a, r, p,  right_prop, S_0 = None,  initial_s_dist = "even", num_rep = 1000, pi_s_a = None, Q = None, epsilon = 0 ):
    # cal real Q function and V function
    Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a) if Q_approximation == None else Iterative_Cal_Q.cal_Q_val_approx_linear_interpolation(p, Q_0, r, num_iter, gamma, S_0,  n_s, n_a)
    V_real, _ = get_V_from_Q(Q_real, n_s, n_a)
    if initial_s_dist == "even":
        R_real = np.mean(V_real)
        initial_w = np.ones(n_s)/n_s
    cov_bools_Q = np.zeros(n_s * n_a)
    cov_bools_V = np.zeros(n_s)
    cov_bools_R = 0.
    print("Q real is {}".format(Q_real))
    #print("V real is {}".format(V_real))
    #print("R real is {}".format(R_real))
    CI_lens_Q = []
    CI_lens_V = []
    CI_lens_R = []
    for i in range(num_rep):
        #print(i)
        if i%100 == 0:
            print(i)
        Q_n, CI_len_Q, V_n, CI_len_V, R_n, CI_len_R= get_CI(Q_approximation,  S_0, num_data, s_0, num_iter, gamma, Q_0, n_s, n_a, r, p , initial_w, right_prop, pi_s_a = pi_s_a, Q=Q, epsilon =epsilon)
        #print("{} th replication : Q_n is {}, CI len is {}".format(i, Q_n, CI_len))
        cov_bool_Q = np.logical_and( Q_real <= (Q_n+CI_len_Q + numerical_tol), Q_real >= (Q_n-CI_len_Q-numerical_tol))
        cov_bool_V = np.logical_and( V_real <= (V_n+CI_len_V + numerical_tol), V_real >= (V_n-CI_len_V-numerical_tol))
        cov_bool_R = np.logical_and(R_real <= (R_n + CI_len_R + numerical_tol ), R_real >= (R_n - CI_len_R-numerical_tol))
        #print(cov_bool_Q)
        #exit()
        #print(cov_bool)
        cov_bools_Q += cov_bool_Q
        cov_bools_V += cov_bool_V
        cov_bools_R += cov_bool_R
        CI_lens_Q.append(CI_len_Q)
        CI_lens_V.append(CI_len_V)
        CI_lens_R.append(CI_len_R)
    CI_len_Q_mean = np.mean(CI_lens_Q)
    CI_len_V_mean = np.mean(CI_lens_V)
    CI_len_R_mean = np.mean(CI_lens_R)
    CI_len_Q_ci = 1.96 * np.std(CI_lens_Q)/np.sqrt(num_rep)
    CI_len_V_ci = 1.96 * np.std(CI_lens_V)/np.sqrt(num_rep)
    CI_len_R_ci = 1.96 * np.std(CI_lens_R)/np.sqrt(num_rep)

    cov_rate_Q = np.divide(cov_bools_Q , num_rep)
    cov_rate_V = np.divide(cov_bools_V , num_rep)
    cov_rate_R = np.divide(cov_bools_R , num_rep)

    return cov_rate_Q, cov_rate_V, cov_rate_R, CI_len_Q_mean, CI_len_V_mean,CI_len_R_mean, CI_len_Q_ci, CI_len_V_ci, CI_len_R_ci




def main():
    num_rep = 1000
    initial_s_dist = "even"
    Q_approximation = None
    #Q_approximation = "linear_interpolation"
    right_prop = 0.8
    # collect data configuration
    num_data = 100000
    print("num_data is {}, rightprop is {}".format(num_data, right_prop))
    n_s = 5
    print("n_s is {}".format(n_s))
    n_a = 2
    s_0 = 2
    n_s_0 = 11
    S_0 = np.linspace(0, n_s, n_s_0)
    S_0[-1] = n_s - 1
    S_0 = S_0.astype(int)
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
    # one replication of coverage test
    #Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    #V_real = get_V_from_Q(Q_real, n_s, n_a)
    #Q_n, CI_len, V_n = get_CI(collec_data_bool, num_data, s_0, num_iter, gamma, Q_0, n_s, n_a, r, p)
    #print(Q_real)
    # print(V_real)
    #print(Q_n)
    # print(V_n)
    #print(CI_len)
    cov_rate_Q, cov_rate_V, cov_rate_R, CI_len_Q_mean, CI_len_V_mean,CI_len_R_mean, CI_len_Q_ci, CI_len_V_ci, CI_len_R_ci = cal_coverage(Q_approximation, num_data, s_0, num_iter, gamma, Q_0, n_s, n_a, r, p, right_prop, S_0, initial_s_dist, num_rep)
    cov_rate_CI_Q =  1.96* np.sqrt(cov_rate_Q *(1-cov_rate_Q)/num_rep)
    cov_rate_CI_V =  1.96* np.sqrt(cov_rate_V *(1-cov_rate_V)/num_rep)
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