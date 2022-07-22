import pandas as pd
import numpy as np
import seaborn as sns
import os
import acquire
import prepare
import split
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats

alpha = 0.05
fig_size = (10,7)

def create_subplots(quant_cols, single_var=True):
    subplot_dim = find_subplot_dim(quant_cols)
    plots= []
    fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(fig_size[0], fig_size[1]))
    axes = axes.flatten()
    
    for axe in axes:
        plots.append(axe)
    
    print(f'Plots: {plots}')
    return plots, fig


def quantitative_hist_boxplot_describe(training_df, quantitative_col_names,separate=True):
    plots, fig = create_subplots(quantitative_col_names)
    for i, col in enumerate(quantitative_col_names):
        plots[i].hist(training_df[col])
        plots[i].set_xlabel(col)
        plots[i].set_ylabel('count')
    plt.show()
    
    if separate:
        print(type(training_df))
        plots, axes = create_subplots(quantitative_col_names)
        for i, col in enumerate(quantitative_col_names):
            training_df.boxplot(ax=plots[i], column=col)
        plt.show()
    else:    
        training_df.boxplot(column=quantitative_col_names)
        plt.show()

    print(training_df[quantitative_col_names].describe().T)
    

def target_freq_hist_count(training_df, target_col):
    freq_hist = training_df[target_col].hist()
    #print(training_df[target_col].value_counts())
    #plt.show()
    return freq_hist

def odd(num):
    if num % 2 != 0:
        return True
    else:
        return False
    
def even(num):
    return not odd(num)

def find_subplot_dim(quant_col_lst):
    
    # goal: make x 
    # checks if len is even (making 2 rows)
    if even(len(quant_col_lst)):
        length = len(quant_col_lst)
    else:
        length = len(quant_col_lst) + 1
        
    divided_by_2 = int(length/ 2)
    divided_by_other_factor = int(length / divided_by_2)
    subplot_dim = [divided_by_2, divided_by_other_factor]
    
    return subplot_dim

def quant_vs_target_bar(train_df, target_col, quant_col_lst, mean_line=False):
    
    subplot_dim = find_subplot_dim(quant_col_lst)
    
    plots = []
    fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], sharex=True, figsize=(10,5))
    
    axes = axes.flatten()
    
    for axe in axes:
        plots.append(axe)

    for n in range(len(quant_col_lst)):    
        sns.barplot(ax=plots[n], x=train_df[target_col], y =train_df[quant_col_lst[n]])
        
        if mean_line:
            avg = train_df[quant_col_lst[n]].mean()
            plots[n].axhline(avg,  label=f'Avg {train_df[quant_col_lst[n]]}')

def describe_quant_grouped_by_target(training_df, quantitative_col, 
                                     target_col):
    lst_cpy = quantitative_col[:]
    lst_cpy.append(target_col)
    
    print(training_df[lst_cpy].groupby(target_col).describe().T)


def target_subsets(target_col, training_df):
    
    values = training_df[target_col].unique()
    subset_dict= {}
    
    for val in values:
        subset_dict[val] = training_df[training_df[target_col]==val]
        
    return subset_dict

def combinations_of_subsets(target_col, training_df):
    subsets = target_subsets(target_col, training_df)
    combos = list(itertools.combinations(subsets.keys(), 2))
    
    return subsets, combos

def mannshitneyu_for_quant_by_target(target_col, training_df, 
                                    quantitative_col):
    
    predictors = {}
    subsets, combos = combinations_of_subsets(target_col, training_df)
    p_exceeds_alpha = []
        

    for i, pair in enumerate(combos):
        
        #print(f'{pair[0]}/{pair[1]}:' )
        predictors[str(pair)] = []
        for col in quantitative_col:
            #print(subsets[pair[0]][col])
            t, p = stats.mannwhitneyu(subsets[pair[0]][col], 
                                      subsets[pair[1]][col])
            #print(f'{pair[0]}/{pair[1]} {col}:')
            #print(f't: {t}, p: {p}\n')
            
            if p < alpha:
                predictors[str(pair)].append({col: [t, p]})
            else:
                p_exceeds_alpha.append([str(pair), col, t, p])
                
                
    return subsets, predictors, p_exceeds_alpha, combos
            
    
def print_mannswhitneyu_predictors(predictors):
    for keys, values in predictors.items():
        print(keys)
        for value in values:
            print(value)
        print()
    
def print_mannswhitneyu_failures(p_exceeds_alpha):
    for val in p_exceeds_alpha:
        print(f'Combination: {val[0]}')
        print(f'Measurement: {val[1]}')
        print(f't: {val[2]}, p: {val[3]}')
        print()
        

def print_quant_by_target(target_col, training_df, quant_col):
    subsets, predictors, p_exceeds_alpha, combos = mannshitneyu_for_quant_by_target(target_col, 
                                                                            training_df, 
                                                                            quant_col)
    print_mannswhitneyu_predictors(predictors)
    print_mannswhitneyu_failures(p_exceeds_alpha)
    
    combo_predic = {}
    for combo in combos:
        combo_predic[combo] = []
        #print(predictors[str(combo)])
        for predic in predictors[str(combo)]:
              #print(list(predic.keys())[0])
              combo_predic[combo].append(list(predic.keys())[0])

    return subsets, predictors, p_exceeds_alpha, combo_predic

def two_quants_by_target_var(target_col, training_df, combo_predic, 
                         subtitle=""):
    
    for combo in combo_predic.keys():
        subplot_dim= find_subplot_dim(combo_predic[combo])
        
        plots = []
        fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(fig_size[0],fig_size[1]))
        
        axes = axes.flatten()
    
        for axe in axes:
            plots.append(axe)
                
        predictors_comb = list(itertools.combinations(combo_predic[combo], 2))
    
        for i, pair in enumerate(predictors_comb):
            sns.scatterplot(x=training_df[pair[0]], y=training_df[pair[1]],
                           hue=training_df[target_col],
                           ax= plots[i])
            plots[i].set_xlabel(pair[0])
            plots[i].set_ylabel(pair[1])
        plt.show()

def categorical_comparisons(train_df, categories, target_var):
    categories_comb = list(itertools.combinations(categories, 2))
    churn_related = [comb for comb in categories_comb if target_var in comb]

    result_dicts = {}
    for comb in churn_related:
        observed = pd.crosstab(train_df[comb[0]], train_df[comb[1]])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        result_dicts[comb] = [chi2, p, degf, expected]

    return result_dicts

def filter_category_compar_results(train_df, categories, target_var):
    results = categorical_comparisons(train_df, categories, target_var)
    filtered = {}

    for key, values in results.items():
        if values[1] < alpha and values[1] != 0:
            filtered.update({key: values})

    sorted_target = []
    for item in filtered.keys():
        item = list(item)
        item.remove(target_var)
        sorted_target.append(item[0])

    print(f"Categories related to {target_var}:")
    for elem in sorted_target:
        print(elem)
    
    return sorted_target

def dataset_reduction(train_df, target_var, categories, quant_cols):
    cats_related_to_target = filter_category_compar_results(train_df, categories, target_var)
    final_df = ['customer_id']
    for col in quant_cols:
        final_df.append(col)
    for cat in cats_related_to_target:
        final_df.append(cat)
    final_df.append(target_var)

    final_df = train_df[final_df]

    return final_df


def overview(train_df, categories,
             quant_cols, target_var):
    quantitative_hist_boxplot_describe(train_df, quant_cols,separate=True)
    target_freq_hist_count(train_df, target_var)
    quant_vs_target_bar(train_df, target_var, quant_cols, mean_line=True)
    describe_quant_grouped_by_target(train_df, quant_cols, target_var)

    subsets, predictors, p_exceeds_alpha, combos = print_quant_by_target(target_var, train_df, quant_cols)
    two_quants_by_target_var(target_var, train_df, combos)
    
    sns.pairplot(train_df, hue=target_var)
    final_df = dataset_reduction(train_df,
                                target_var,
                                categories,
                                quant_cols)

    return final_df