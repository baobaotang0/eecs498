import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    '''
    (c) Implement this Monte Carlo estimator (e.g., in MATLAB, Python) and use it to estimate
    the value for π. Perform this for a range of sample sizes n (e.g., for n = 101,102,103,...). On
    the same figure, plot with respect to n the following: (i) the absolute error |π − ̄sn|, (ii) the
    standard error estimated from the same samples, and (iii) a reference line with slope −1/2
     in the log-log scale.
    '''
    # a = 1
    # n_list = [n for n in range(100, 10000, 100)]
    # abs_error = [0] * len(n_list)
    # standard_error = [0] * len(n_list)
    # for j in range(len(n_list)):
    #     n = n_list[j]
    #     p = np.random.rand(n, 2) * a
    #     indicator = np.zeros((n, 1))
    #     for i in range(n):
    #         if p[i, 0] ** 2 + p[i, 1] ** 2 <= a ** 2:
    #             indicator[i] = 1
    #     Sn_hat = np.count_nonzero(indicator) / n * 4
    #     abs_error[j] = np.absolute(np.pi-Sn_hat)
    #     standard_error[j] = 4*np.std(indicator)/np.sqrt(n)
    # plt.loglog(n_list, abs_error)
    # plt.loglog(n_list, standard_error)
    # plt.loglog([n_list[0], n_list[-1]], [standard_error[0],
    #                                      np.exp(-0.5*np.log(n_list[-1]) + np.log(standard_error[0])--0.5*np.log(n_list[0]))])
    # plt.legend(["abs_error VS n", "standard_error VS n", "a reference line with slope −1/2"])
    # plt.show()

    '''
    (d) Choose a fixed n, and repeat your Monte Carlo estimate of π many (say m) times. Make a
    histogram of errors π −  ̄sn (no absolute value) for these m trials. What is the relationship
    between the standard deviation of this histogram of error values, compared to the standard
    error?
    '''
    a = 1
    m = 1000
    n = 1000
    error = [0] * m
    standard_deviation = [0] * m
    standard_error = [0] * m
    Sn_hat = [0] * m
    for j in range(m):
        p = np.random.rand(n, 2) * a
        indicator = np.zeros((n, 1))
        for i in range(n):
            if p[i, 0] ** 2 + p[i, 1] ** 2 <= a ** 2:
                indicator[i] = 1
        Sn_hat[j] = np.count_nonzero(indicator) / n * 4
        error[j] = np.pi-Sn_hat[j]
        standard_deviation[j] = 4*np.std(indicator)
        standard_error[j] = standard_deviation[j]/np.sqrt(n)
    hist = plt.hist(error)
    plt.title("error")
    plt.show()
    plt.hist(standard_error)
    plt.title("standard_error")
    plt.show()
    std_abs_error = np.std(error)
    std_std_error = 4*np.std(Sn_hat)/np.sqrt(n)
    print(std_abs_error, std_std_error)
