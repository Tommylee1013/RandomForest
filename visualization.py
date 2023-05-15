import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sklearn import metrics
import seaborn as sns

def plot_calibration_curve(y_true, y_prob, n_bins=10, ax=None, hist=True, normalize=False):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=normalize)
    if ax is None:
        ax = plt.gca()
    if hist:
        ax.hist(y_prob, weights=np.ones_like(y_prob) / len(y_prob), alpha=.4,
               bins=np.maximum(10, n_bins))
    ax.plot([0, 1], [0, 1], ':', c='k')
    curve = ax.plot(prob_pred, prob_true, marker="o")

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("fraction of positive samples")
    ax.set(aspect='equal')
    return curve

def plotMetrics(X_test, y_test, fit, ax = None, hist = True) :
    y_pred_rf = fit.predict_proba(X_test)[:, 1]
    y_pred = fit.predict(X_test)
    fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred_rf)
    #print(metrics.classification_report(y_test, y_pred, target_names = ['no trade',' trade']))

    if hist :
        ax = sns.displot(fit.predict_proba(X_test), height=4, aspect=2, alpha=0.4, kde=True)

    ax.set_xlabels('Probability')
    ax.set_ylabels('Counts')

    return ax

    #plt.figure(figsize = (9,6))
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(fpr_rf, tpr_rf, label = 'Random Forest Classifier')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.title('ROC curve')
    #plt.legend(loc='best')
    #plt.show()