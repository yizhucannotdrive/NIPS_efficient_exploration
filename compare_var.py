import numpy as np
import Iterative_Cal_Q
import inference
import matplotlib.pyplot as plt

def transition_mat_S (p, right_prop, n_s, n_a):
    tran_M = np.zeros((n_s,n_s))
    for i in range(n_s):
        for j in range(n_s):
            tran_M[i,j] = (1-right_prop) * p[i*n_a* n_s + 0 * n_s + j] + right_prop * p[i*n_a* n_s + 1 * n_s + j]
    return tran_M

def transition_mat_S_A (p, right_prop, n_s, n_a):
    tran_M = np.zeros((n_s * n_a ,n_s * n_a))
    for i in range(n_s):
        for j in range(n_a):
            for k in range(n_s):
                for l in range(n_a):
                    #tran_M[i*n_a + j , k*n_a + l] = stat_p[i] * ((2*right_prop-1)*j + 1-right_prop) * p[i*n_a* n_s + j * n_s + k] * ((2*right_prop-1)*l + 1-right_prop)
                    tran_M[i*n_a + j , k*n_a + l] =  p[i*n_a* n_s + j * n_s + k]  * ((2*right_prop-1)*l + 1-right_prop)
    return tran_M
def solveStationary( A ):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.linalg.lstsq( a, b, rcond = None )[0]
    #return np.linalg.lstsq( a, b )[0]


def main():
    #collect data with random policy
    n_s = 6
    n_a = 2
    p = np.zeros(n_s * n_a * n_s)
    r = np.zeros(n_s * n_a)
    r[0] = 0.1
    r[-1] = 10
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
    initial_w = np.ones(n_s)/n_s
    Q_0 = np.zeros(n_s * n_a)
    num_iter = 1000
    gamma = 0.95
    num_data = 1000000
    Q_real = Iterative_Cal_Q.cal_Q_val(p, Q_0, r, num_iter, gamma, n_s, n_a)
    V_real, V_max_index = inference.get_V_from_Q(Q_real, n_s, n_a)
    R_real = np.mean(V_real)
    right_props = np.linspace(0.8,0.9,10, endpoint = False)
    CI_len_Rs = []
    for right_prop in right_props:
        var_r = np.zeros(n_s * n_a)
        tran_M_S_A = transition_mat_S_A (p, right_prop, n_s, n_a)
        f = solveStationary(tran_M_S_A)
        f = np.array(f).reshape(-1, )
        #print(tran_M_S_A)
        #print(f)
        Sigma_Q, Sigma_V, Sigma_R = inference. cal_Sigma_n(p, f, var_r, V_real, gamma, n_s, n_a, V_max_index, initial_w)
        CI_len_R = 1.96 * np.sqrt(Sigma_R) / np.sqrt(num_data)
        #print(CI_len_R)
        CI_len_Rs.append(CI_len_R)
    print(CI_len_Rs)
    plt.plot(right_props, CI_len_Rs, 'ro--', markersize=6)
    plt.xlabel("exploration right decision probability")
    plt.ylabel("CI_len")
    plt.title("Compare variance of different exploration strategy")
    plt.show()


if __name__ == "__main__":
    main()