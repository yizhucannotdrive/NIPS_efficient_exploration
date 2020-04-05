import numpy as np
import pickle
def get_right_prop(s,pi_s_a, n_a):
    if np.sum(pi_s_a[s*n_a: (s+1) *n_a])!=0:
        return pi_s_a[s*n_a +1]/np.sum(pi_s_a[s*n_a: (s+1) *n_a])
    else:
        return 0.5
def get_right_prop_Q(epsilon, Q, s, n_a):
    right_pro = 1- epsilon if Q[s * n_a +1] >= Q[s * n_a] else epsilon
    return right_pro
def collect_data(p, r, num_data, s_0, n_s, n_a, right_prop = 0.85, save_data = False, pi_s_a = None, Q = None, epsilon = 0.01, print_pro_right =False, std = 1):
    #print(std)
    data = []
    s = s_0
    for i in range(num_data):
        if not (pi_s_a is None):
            right_prop = get_right_prop(s, pi_s_a, n_a)
        if not (Q is None):
            right_prop = get_right_prop_Q(epsilon, Q, s, n_a)
        a = np.random.binomial(1, right_prop, 1)[0]
        p_sa = p[(s * n_a * n_s + a * n_s): (s * n_a * n_s + (a + 1) * n_s)]
        s_new = int(np.random.choice(n_s, 1, p=p_sa)[0])
        if print_pro_right:
            print("%%%%")
            print(s)
            print(right_prop)
            print(p_sa)
            print(s_new)
            print(std)
            print("%%%%%")
        r_sa = np.random.normal(r[s * n_a + a], std)
        #r_sa = np.random.normal(r[s * n_a + a],n_s-s)
        #print("#", r_sa)
        data_i = (int(s), a, r_sa, s_new)
        # print(data_i)
        s = np.copy(s_new)
        data.append(data_i)
    if save_data:
        pickle.dump(data, open("data", "wb"))
    return data

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
    num_data = 1000000
    s_0 = 2
    data = collect_data(p, r, num_data, s_0, n_s, n_a)


if __name__ == "__main__":
    main()