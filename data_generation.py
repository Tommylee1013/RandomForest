import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm


# initialization 
prob_switch = 0.2
stdev = 0.012623971978355785 # S&P500 전구간 stdev
R1, R2, R3 = 0.012, -0.009, 0.006 # return init value
INNER_STEPS = 10

def _gen_data(phi1, phi2, phi3, flag, stdev, drift, steps):
    # Initial values for lagged returns
    r1, r2, r3 = R1, R2, R3
    # Create data set based on AR(p)
    rets, flags = [], []
    for _ in range(0, steps):
        rt = drift + phi1*r1 + phi2*r2 + phi3*r3 + np.random.normal(loc=0, scale=stdev, size=1)
        flags.append(flag)
        rets.append(float(rt))
        r3, r2, r1 = r2, r1, rt
    return rets, flags


def _gen_dual_regime(phi, steps, inner_steps, prob_switch, stdev):
    rets, flags, is_regime_two = [], [], 0
    for _ in range(0, steps):
        is_regime_two =  np.random.uniform() < prob_switch

        if is_regime_two:
            rets_regime, flags_regime = _gen_data(phi1=-phi[0], phi2=-phi[1], phi3=-phi[2],
                                                  flag=1, steps=inner_steps,
                                                  stdev=stdev, drift=-0.0001)
        else:
            rets_regime, flags_regime = _gen_data(phi1=phi[0], phi2=phi[1], phi3=phi[2],
                                                  flag=0, steps=inner_steps,
                                                  stdev=stdev, drift=0.0001)
        rets.extend(rets_regime)
        flags.extend(flags_regime)
    return rets, flags


def dual_regime(total_steps, phi, prob_switch=prob_switch, stdev=stdev):
    # Params
    inner_steps = INNER_STEPS
    steps = int(total_steps / inner_steps)  # Set steps so that total steps is reached

    # Gen dual regime data
    rets, flags = _gen_dual_regime(phi = phi, steps=steps, inner_steps=inner_steps,
                                   prob_switch=prob_switch, stdev=stdev)

    # Convert to DF
    date_range = pd.date_range(end=dt.datetime.now(),
                               periods=steps * inner_steps,
                               freq='d', normalize=True)
    
    data = pd.DataFrame({'rets': np.array(rets).flatten(), 'flags': flags}, index=date_range)
    return data, get_SNR(data)


def prep_data(data, with_flags=True):
    # Set target variable
    data['target'] = data['rets'].apply(lambda x: 0 if x < 0 else 1).shift(-1)  # Binary classification
    data['target_rets'] = data['rets'].shift(-1)  # Add target rets for debugging
    data.dropna(inplace=True)

    # Auto-correlation trading rule: trade sign of previous day.
    data['pmodel'] = data['rets'].apply(lambda x: 1 if x > 0.0 else 0)

    # Strategy daily returns
    data['prets'] = (data['pmodel'] * data['target_rets']).shift(1)  # Lag by 1 to remove look ahead and align dates
    data.dropna(inplace=True)

    # Add lag rets 2 and 3 for Logistic regression
    data['rets2'] = data['rets'].shift(1)
    data['rets3'] = data['rets'].shift(2)
    
    # Add Regime indicator if with_flags is on
    if with_flags:
        data['regime'] = data['flags'].shift(5)

    model_data = data[data['pmodel'] == 1]
    return model_data.dropna(), data


def get_SNR(data):
    data = data['rets']
    data = pd.concat([data, data.shift(1), data.shift(2), data.shift(3)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    data = sm.tools.tools.add_constant(data)
    data.columns = ['Const', 'ret', 'ret(1)', 'ret(2)', 'ret(3)']
    model = sm.OLS(data['ret'], data.loc[:,['Const', 'ret(1)', 'ret(2)', 'ret(3)']])
    results = model.fit()
    return results.rsquared


def classification_stats(actual, predicted, prefix, get_specificity):
    # Create Report
    report = classification_report(actual, predicted, output_dict=True,
                                   labels=[0, 1], zero_division=0)
    # Extract (long only) metrics
    report['1'][prefix + '_accuracy'] = report['accuracy']
    report['1'][prefix + '_auc'] = roc_auc_score(actual, predicted)
    report['1'][prefix + '_macro_avg_f1'] = report['macro avg']['f1-score']
    report['1'][prefix + '_weighted_avg'] = report['weighted avg']['f1-score']

    # To DataFrame
    row = pd.DataFrame.from_dict(report['1'], orient='index').T
    row.columns = [prefix + '_precision', prefix + '_recall', prefix + '_f1_score',
                    prefix + '_support', prefix + '_accuracy', prefix + '_auc',
                    prefix + '_macro_avg_f1', prefix + '_weighted_avg_f1']

    # Add Specificity
    if get_specificity:
        row[prefix + '_specificity'] = report['0']['recall']
    else:
        row[prefix + '_specificity'] = 0

    return row