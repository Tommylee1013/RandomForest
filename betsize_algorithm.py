def de_prado_bet_size(prob_series, clip=True):
    # Can't compute for p = 1 or p = 0, leads to inf.
    p = prob_series.copy()
    p[p == 1] = 0.99999
    p[p == 0] = 0.00001
    # Getting max value from training set
    num_classes = 2
    dp_sizes = (p - 1 / num_classes) / ((p * (1 - p)) ** 0.5)
    dp_t_sizes = dp_sizes.apply(lambda s: norm.cdf(s))
    dp_bet_sizes = dp_t_sizes
    # no sigmoid function, only clipping?
    dp_bet_sizes[dp_bet_sizes < 0.5] = 0
    return dp_bet_sizes

def check_stats(rets):
    if np.std(rets) == 0.0:
        stdev = 10000
    else:
        stdev = np.std(rets)

    if (np.mean(rets) <= 0.00001) and (np.mean(rets) >= -0.00001):
        mean = -10000
    else:
        mean = np.mean(rets)
    
    return mean, stdev

def target_linear(x):
    # Linear function
    f = lambda p: min(max(x[0] * p + x[1], 0), 1)
    f = np.vectorize(f)
    # Backtest
    rets = f(prob_train[prob_train > 0.5]) * target_train_p[prob_train > 0.5]
    # Solve for no positions taken
    mean, stdev = check_stats(rets)
    # Sharpe Ratio
    sr = mean / stdev
    return -sr