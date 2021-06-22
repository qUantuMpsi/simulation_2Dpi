import scipy.integrate as sci
import numpy as np
import matplotlib.pyplot as plt

p = 1.1  # sigma_p
c = 12.  # sigma_c
s = 2.5 * c  # sigma_s
N = 31  # dimension of matrix is N*N
es = np.arange(0, 1, 0.1)  # values of epsilon


def f(e, qs, qi):  # function for JTMA depending on qs qi and epsilon
    cs = abs(np.sqrt(1 / (1 - e)))
    qtp = cs * abs(qs + qi) / np.sqrt(2)
    qtm = cs * abs(qs - qi) / np.sqrt(2)
    return np.exp(- (abs(qs + qi) ** 2) / (2 * p ** 2)) * np.sinc(((2 + e) * qtm ** 2 + e * qtp ** 2) / (s ** 2))


def C(q):  # collection mode function
    return ((np.sqrt(np.pi) * c) ** (-1. / 2.)) * np.exp(- (abs(q) ** 2) / (2 * c ** 2))


def G(qs, qi, e):  # collected JTMA function
    return 50 * (c ** -1.) * np.exp(- (abs(qs) ** 2 + abs(qi) ** 2) / (2 * c ** 2)) * f(e, qs, qi)


Pr1 = np.identity(N)  # initialising G matrix

for e in es:  # loop starts for different e value
    for i in range(N):  # loop for different a_s value
        for j in range(N):  # loop for different a_i value
            a_s = i - (N - 1) / 2
            a_i = j - (N - 1) / 2
            Pr1[i][j] = G(a_s, a_i, e)
            print(i, j, Pr1[i][j])

    x = np.arange(-(N - 1) / 2, 1 + (N - 1) / 2)
    y = np.arange(-(N - 1) / 2, 1 + (N - 1) / 2)
    plt.contourf(x, y, Pr1)
    plt.title('JTMA $\epsilon$ = %.2f' % e)
    plt.show()  # visualising G matrix which is collected JTMA

    Pr = np.identity(N)  # initialisng for 2Dpi measurement matrix.

    for i in range(N):
        for j in range(i, N):
            a_i = i - (N - 1) / 2
            a_s = j - (N - 1) / 2
            I1 = sci.dblquad(G, -np.inf, a_s, -np.inf, a_i, args=[e])[0]  # G--
            I2 = sci.dblquad(G, a_s, np.inf, a_i, np.inf, args=[e])[0]  # G++
            I3 = sci.dblquad(G, -np.inf, a_s, a_i, np.inf, args=[e])[0]  # G-+
            I4 = sci.dblquad(G, a_s, np.inf, -np.inf, a_i, args=[e])[0]  # G+-
            if i == j:
                Pr[i][j] = 0.5 * abs(I1 + I2 - I3 - I4) ** 2
            else:
                Pr[i][j] = abs(I1 + I2 - I3 - I4) ** 2
            print(i, j, Pr[i][j])

    Pr = Pr + Pr.transpose()  # for symmetric Pr
    x = np.arange(-(N - 1) / 2, 1 + (N - 1) / 2)
    y = np.arange(-(N - 1) / 2, 1 + (N - 1) / 2)
    plt.contourf(x, y, Pr)
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('Pr($a_s$, $a_i$) $\epsilon$ = %.2f' % e)
    plt.show()