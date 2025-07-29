from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse 
import scipy
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix


from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_reg_plot(xs, ys, dataset_name, feature_name_x, feature_name_y, feature_dim, ransac_th=10, savePath=None, figureSize=(10, 7), dpi=72, figureFormat='png'):
    """
    xs: data series
    ys: target values (predictions)
    dataset_name: name of the dataset to include in plot axis and title
    """
    
    estimators = [('OLS', LinearRegression()),
                  ('Theil-Sen', TheilSenRegressor(random_state=42)),
                  ('RANSAC', RANSACRegressor(base_estimator=LinearRegression(),
                             min_samples=5, max_trials=100,
                             loss='absolute_error',random_state=42,residual_threshold=ransac_th))]

    colors = {'OLS': 'red', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen'}
    lw = 2


    xs = xs.astype('float32').values

    ys = ys.astype('float32').values

    x = np.arange(max(xs.max(),ys.max()))
    
    Xs = xs[:, np.newaxis]
    line_x = np.array([0, max(xs.max(),ys.max())])
    
    fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)
    plt.rcParams.update({'font.size': 16})
    
    for name, estimator in estimators:
        estimator.fit(Xs, ys)
        y_pred = estimator.predict(line_x.reshape(2, 1))

        R2_t = r2_score(ys, estimator.predict(Xs))

        r_squared = estimator.score(Xs, ys)
        # Estimated coefficients
        try:
            m, c = float(estimator.coef_), float(estimator.intercept_)
        except:
            m, c = float(estimator.estimator_.coef_), float(estimator.estimator_.intercept_)

        RMSE = np.sqrt(mse(ys, estimator.predict(Xs)))
        
    
        ax.plot(line_x, y_pred, color=colors[name], linewidth=lw,
                 label='%s r2: %.2f m: %.2f c: %.2f mse: %.2f' % (name, r_squared, m, c, RMSE))
    ax.scatter(xs, ys, color='#003F72', s=50,alpha = 0.5)
    ax.axis('tight')
    ax.legend(loc=0)
    ax.set_xlabel(f'{feature_name_x} ({feature_dim})',fontsize=18)
    ax.set_ylabel(f'{feature_name_y} ({feature_dim})',fontsize=18)

    if (savePath is not None):
        fig.savefig(savePath, format=figureFormat, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plt_CM(data, feature_method1, feature_method2, ylabel, xlabel, title, cm_binary=True, savePath=None, figureSize=(10, 7),  dpi=300, figureFormat='png'):
    
    """
    Plot confusion matrix for evaluating the model's performance. 
    data: categorical dataframe containing target values (from clinical reports) and predicted values by AI Pipeline model. 
    feature_method1: column name for target values (str)
    feature_method1: column name for predicted values (str)
    ylabel: label for y-axis (str) 
    xlabel: label for x-axis (str)
    title: title for the figure
    savePath: path to save the image (optional)

    Example: plt_CM(df_LV,'LVEF_echo_cat','LVEF_model_cat','Echo','AI Pipeline', 'LVEF', True, figureFormat=figFormat, savePath=path_)

    """

    labels = ['normal','mildly abnormal','moderately abnormal','severely abnormal']
    severity_dict = {'normal': 0, 'mildly abnormal': 1, 'moderately abnormal': 2, 'severely abnormal': 3}

    np.set_printoptions(precision=2)
    data.replace(severity_dict, inplace=True)
    ys = data[feature_method1]
    xs = data[feature_method2]
    
    if cm_binary:
        labels = ['N','Ab']
        severity_dict = {'Normal': 0, 'Abnormal': 1}
        np.set_printoptions(precision=2)
        data.replace(severity_dict, inplace=True)
        ys = (data[feature_method1].astype(int) > 0).astype(int)
        xs = (data[feature_method2].astype(int) > 0).astype(int)
    
    cm = confusion_matrix(ys.astype('category'), xs.astype('category'))
    accuracy = np.trace(cm) / np.sum(cm).astype('float')   ## total samples that were correctly classified
    if cm_binary:
        auc = roc_auc_score(ys.astype(int), (xs.astype(int)))
        f1 = f1_score(ys.astype(int), (xs.astype(int)))
    else:
        ys_onehot = pd.get_dummies(ys)
        xs_onehot = pd.get_dummies(xs)
        auc = roc_auc_score(ys_onehot, xs_onehot, multi_class='ovo')
        f1 = f1_score(ys.astype(int), (xs.astype(int)), average='weighted')

    misclass = 1 - accuracy
    k = cohen_kappa_score(ys.astype(int), (xs.astype(int)))
    mean_absolute_error = mae(ys.astype(int), (xs.astype(int)))
    plt.rcParams.update({'font.size': 16})

    figsize = (6, 6) if cm_binary else (12, 12)

    fig, ax = plot_confusion_matrix(conf_mat=cm,colorbar=False,
                                    show_absolute=True,
                                    show_normed=True,
                                    class_names=labels,
                                    figsize=figsize)

    ax.set_ylabel(ylabel,fontsize=18)
    ax.set_xlabel(f'{xlabel}',fontsize=18)

    ax.set_title(r"$\bf{"+title+"}$" + f" (acc={accuracy:.2f}; kappa={k:.2f}; f1={f1:.2f}; auc={auc:.2f})", fontsize=16)
    
    if savePath:
        fig.savefig(savePath,bbox_inches="tight", format=figureFormat, dpi=1200)
    else:
        plt.show()

    return accuracy, auc, k, f1, mean_absolute_error


def get_stats(df, list_of_features):
    """
    Compute stats between two values. Stats are: mannwhitneyu, Spearmanr correlation, coefficient of variation, mean_absolute_error, mean_squared_error
    Input: df - dataframe of features
            list_of_features - Features to compare. They must be in pairs in the order of comparison. 
    """

    df_stats  = df[list_of_features].agg(['mean','median','std','min', 'max'])
    
    df_stats.T.plot(kind = "barh", y = ["mean"], legend = False,
                fontsize=14,  xerr='std', capsize=4,rot=45) #title = 'LA - volumetrics'
    plt.show()
    j=0
    for i in range(round(len(list_of_features)-len(list_of_features)/2)):
        df_stats = df.dropna(subset=[list_of_features[j],list_of_features[j+1]])
        print('size of data:',df_stats.shape[0])
        H, p = mannwhitneyu(df_stats[list_of_features[j]].dropna().values, df_stats[list_of_features[j+1]].dropna().values,alternative='two-sided')
        print(f'p-value(MWU) between {list_of_features[j]} and {list_of_features[j+1]}: {p:.1e}')
        H, p = scipy.stats.spearmanr(df_stats[list_of_features[j]].dropna().values, df_stats[list_of_features[j+1]].dropna().values)
        print(f'Spearmanr correlation between {list_of_features[j]} and {list_of_features[j+1]}: {p:.1e}')
        
        mae_ = mae(df_stats[list_of_features[j]].dropna().values, df_stats[list_of_features[j+1]].dropna().values)
        print(f'mean_absolute_error between {list_of_features[j]} and {list_of_features[j+1]}: {mae_:.2f}')
        mse_ = mse(df_stats[list_of_features[j]].dropna().values, df_stats[list_of_features[j+1]].dropna().values)
        print(f'mean_squared_error between {list_of_features[j]} and {list_of_features[j+1]}: {mse_:.2f}')
        cov1, cov2 = scipy.stats.variation(df_stats[list_of_features[j]])*100, scipy.stats.variation(df_stats[list_of_features[j+1]])*100
        print(f'coefficient of variation of {list_of_features[j]}: {cov1:.2f} and {list_of_features[j+1]}: {cov2:.2f}')
        j += 2
        print()
