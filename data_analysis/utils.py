# data_analysis/utils.py
import pandas as pd
import numpy as np
import json

def load_data(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def remove_outliers(df, column, z_threshold=3.0):
    """
    Remove outliers from a DataFrame based on z-score of a specific column.
    
    Args:
        df (pandas.DataFrame): DataFrame to filter
        column (str): Column name to check for outliers
        z_threshold (float): Z-score threshold for outlier detection
        
    Returns:
        pandas.DataFrame: DataFrame with outliers removed
    """
    # Calculate z-scores
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    
    # Keep rows where z-score is below threshold
    filtered_df = df[z_scores < z_threshold].copy()
    
    # Report number of outliers removed
    outliers_count = len(df) - len(filtered_df)
    if outliers_count > 0:
        print(f"Removed {outliers_count} outliers from '{column}' column using z-score threshold of {z_threshold}")
        print(f"Max value before removal: {df[column].max():.2f}, after removal: {filtered_df[column].max():.2f}")
    
    return filtered_df

def preprocess_data(data):
    """Process raw data into dataframes for analysis."""
    # Create main metrics dataframe
    metrics_df = pd.DataFrame([
        {
            'methodology': repo['pm_classification']['primary_methodology'],
            'stars': repo['stars'],
            'forks': repo['forks'],
            'issues_open': repo['issues'],
            'issue_closure_rate': repo['lifecycle_indicators']['community_health']['issue_closure_rate'],
            'pr_acceptance_rate': repo['lifecycle_indicators']['community_health']['pr_acceptance_rate'],
            'contributors': repo['lifecycle_indicators']['community_health']['unique_contributors'],
            'growth_rate': repo['lifecycle_indicators']['growth_trajectory']['star_growth_rate'],
            'continuity_score': repo['lifecycle_indicators']['development_continuity']['continuity_score']
        }
        for repo in data
    ])

    # Remove outliers from growth_rate
    metrics_df = remove_outliers(metrics_df, 'growth_rate')
    
    # Create interaction data
    interaction_data = pd.DataFrame([
        {
            'team_interaction': repo['pm_classification']['raw_scores']['kanban'] / 
                               sum(repo['pm_classification']['raw_scores'].values()),
            'active_participation': repo['pm_classification']['raw_scores']['agile_scrum'] /
                                   sum(repo['pm_classification']['raw_scores'].values()),
            'issue_closure_rate': repo['lifecycle_indicators']['community_health']['issue_closure_rate'],
        }
        for repo in data
    ])
    
    # Create community metrics
    community_df = pd.DataFrame([
        {
            'methodology': repo['pm_classification']['primary_methodology'],
            'issue_closure_rate': repo['lifecycle_indicators']['community_health']['issue_closure_rate'],
            'pr_acceptance_rate': repo['lifecycle_indicators']['community_health']['pr_acceptance_rate'],
            'response_time': repo['lifecycle_indicators']['community_health']['avg_issue_response_time'],
            'contributors': repo['lifecycle_indicators']['community_health']['unique_contributors'],
            'contributor_growth': repo['lifecycle_indicators']['community_health']['contributor_growth']
        }
        for repo in data
    ])
    # Create dataframe with lifecycle information
    lifecycle_df = pd.DataFrame([
        {
            'methodology': repo['pm_classification']['primary_methodology'],
            'development_stage': repo['lifecycle_indicators']['development_continuity']['development_stage'],
            'activity_pattern': repo['lifecycle_indicators']['activity_pattern'],
            'issue_closure_rate': repo['lifecycle_indicators']['community_health']['issue_closure_rate'],
            'stars': repo['stars']
        }
        for repo in data
    ])

    # Create time-based metrics
    growth_df = pd.DataFrame([
        {
            'methodology': repo['pm_classification']['primary_methodology'],
            'age_days': (pd.to_datetime('2025-03-05', utc=True) - pd.to_datetime(repo['created_at'])).days,
            'star_growth_rate': repo['lifecycle_indicators']['growth_trajectory']['star_growth_rate'],
            'fork_growth_rate': repo['lifecycle_indicators']['growth_trajectory']['fork_growth_rate'],
            'stars': repo['stars'],
            'forks': repo['forks']
        }
        for repo in data
    ])

    # Remove outliers from growth rates in growth_df
    growth_df = remove_outliers(growth_df, 'star_growth_rate')
    growth_df = remove_outliers(growth_df, 'fork_growth_rate')
    
    return metrics_df, interaction_data, community_df, lifecycle_df, growth_df