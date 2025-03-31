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


def regression_analysis(data, dependent_var, independent_vars, output_dir=None):
    """
    Perform simple linear regression analysis and save results to a file.
    
    Args:
        data (pd.DataFrame): Input dataframe containing all variables
        dependent_var (str): Name of the dependent variable column
        independent_vars (list): List of independent variable column names
        output_dir (str or Path): Directory to save results, if None returns results dict
    
    Returns:
        dict: Dictionary containing regression results
    """
    try:
        # Prepare the data
        X = data[independent_vars].fillna(0)  # Handle missing values
        y = data[dependent_var].fillna(0)
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit regression model
        model = sm.OLS(y, X).fit()
        
        # Extract key metrics
        results = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'p_values': model.pvalues.to_dict(),
            'coefficients': model.params.to_dict(),
            'num_observations': model.nobs
        }
        
        if output_dir:
            from datetime import datetime
            import os
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results tables
            summary_data = {
                'Metric': ['R-squared', 'Adjusted R-squared', 'Number of Observations'],
                'Value': [
                    f"{results['r_squared']:.4f}",
                    f"{results['adj_r_squared']:.4f}",
                    f"{results['num_observations']:.0f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            coefficients_data = {
                'Variable': list(results['coefficients'].keys()),
                'Coefficient': [f"{v:.4f}" for v in results['coefficients'].values()],
                'P-value': [f"{results['p_values'][k]:.4f}" for k in results['coefficients'].keys()]
            }
            coefficients_df = pd.DataFrame(coefficients_data)
            
            # Save to files
            output_file = output_dir / f'regression_analysis_{timestamp}.txt'
            with open(output_file, 'w') as f:
                f.write(f"Regression Analysis Results\n")
                f.write(f"Dependent Variable: {dependent_var}\n")
                f.write(f"Independent Variables: {', '.join(independent_vars)}\n")
                f.write(f"\nSummary Statistics:\n")
                f.write(summary_df.to_string(index=False))
                f.write(f"\n\nCoefficients and P-values:\n")
                f.write(coefficients_df.to_string(index=False))
            
            print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error in regression analysis: {str(e)}")
        return None