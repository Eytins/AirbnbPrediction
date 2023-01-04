import pandas as pd

params = pd.read_csv('final_features.csv')
params.boxplot()

def boxplot_fill(col):
    iqr = col.quantile(0.75) - col.quantile(0.25)
    u_th = col.quantile(0.75) + 1.5 * iqr
    l_th = col.quantile(0.25) - 1.5 * iqr
    def box_trans(x):
        if x > u_th:
            return u_th
        elif x < l_th:
            return l_th
        else:
            return x
    return col.map(box_trans)
boxplot_fill(params[target]).hist()

