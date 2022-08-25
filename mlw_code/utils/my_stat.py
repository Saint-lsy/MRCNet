from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def get_sub_df(df, name_list, name_col):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df) if df.endswith('xlsx') else pd.read_csv(df)
    df = df[df[name_col].isin(name_list)]
    return df


def get_num_table(values_1, values_2):
    count_1 = Counter(values_1)
    count_2 = Counter(values_2)
    num_table = {}
    for key in np.unique(np.append(list(count_1.keys()), list(count_2.keys()))):
        try:
            key = int(key)
        except:
            pass
        num_table.update({key: [count_1[key], count_2[key]]})
    # print(num_table)
    return num_table

def clinic_stat(df_1, df_2, discrete_variable, continuous_variable):
    for col_name in df_1:
        if not col_name in discrete_variable+continuous_variable:
            continue

        print('Characteristics:', col_name)
        values_1 = df_1[col_name].dropna(axis=0, how='all').values
        values_2 = df_2[col_name].dropna(axis=0, how='all').values
        values_all = np.append(values_1, values_2)

        if col_name in discrete_variable:
            num_table = get_num_table(values_1, values_2)

            p = chi2_contingency(list(num_table.values()))[1]
            print('distribution:', num_table)
            # print('percent: %.3f, %.3f, %.3f, %.3f' % (num_table[0][0] / len(values_all),
            #                                            num_table[0][1] / len(values_all), 
            #                                            num_table[1][0] / len(values_all), 
            #                                            num_table[1][1] / len(values_all)))
            print('p-Value:', p, '   Method: Chi^2')
            print('----------')

        elif col_name in continuous_variable:
            
            p = mannwhitneyu(values_1, values_2)[1]
            print(col_name, 'on Total Set:', np.mean(values_all), '+/-', np.std(values_all))
            print(col_name, 'on df1 Set:', np.mean(values_1), '+/-', np.std(values_1))
            print(col_name, 'on df2 Set: ', np.mean(values_2), '+/-', np.std(values_2))
            print('p-Value:', p, '   Method: Mann-Whitney U-Test')
            print('----------')
    pass


def get_meter(df, target_col, pred_col):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df) if df.endswith('xlsx') else pd.read_csv(df)
    # TP    predict 1 label 1
    TP = ((df[pred_col] == 1) & (df[target_col] == 1)).sum()
    # FP    predict 1 label 0
    FP = ((df[pred_col] == 1) & (df[target_col] == 0)).sum()
    # FN    predict 0 label 1
    FN = ((df[pred_col] == 0) & (df[target_col] == 1)).sum()
    # TN    predict 0 label 0
    TN = ((df[pred_col] == 0) & (df[target_col] == 0)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)  # accuracy
    spec = TN / (TN + FP)     # specificity
    sens = TP / (TP + FN)     # sensitivity
    p = TP / (TP + FP)        # precision  精确率 真阳性比率
    r = TP / (TP + FN)        # recall  召回率
    f1 = 2 * r * p / (r + p)  # F1 socre
    print(sens, spec, p, TN/(TN+FN))

    # print(TP, FP, FN, TN)

# df_pth = r'E:\Desktop\2001_NFYY-LNM\0216 嘉铭多中心数据整理版.xlsx'
# reader_val(df_pth, '淋巴结情况', 'CT报淋巴结转移情况')

