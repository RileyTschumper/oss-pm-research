# data_analysis/statistical_tests.py
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def run_methodology_comparison_tests(metrics_df):
    """Run statistical tests comparing methodologies."""
    results = {}
    
    # T-tests comparing pairs of methodologies
    methodologies = metrics_df['methodology'].unique()
    for i, m1 in enumerate(methodologies):
        for m2 in methodologies[i+1:]:
            group1 = metrics_df[metrics_df['methodology'] == m1]['issue_closure_rate']
            group2 = metrics_df[metrics_df['methodology'] == m2]['issue_closure_rate']
            
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                results[f"{m1}_vs_{m2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
    
    # ANOVA test
    groups = [metrics_df[metrics_df['methodology'] == m]['issue_closure_rate'] 
              for m in methodologies if len(metrics_df[metrics_df['methodology'] == m]) > 0]
    
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        results["ANOVA"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    
    return results

def correlation_analysis(data):
    """Perform correlation analysis on key metrics."""
    correlation_matrix = data.corr()
    return correlation_matrix

def regression_analysis(data, dependent_var, independent_vars):
    """Perform regression analysis."""
    X = data[independent_vars]
    y = data[dependent_var]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit regression model
    model = sm.OLS(y, X).fit()
    
    return model.summary()