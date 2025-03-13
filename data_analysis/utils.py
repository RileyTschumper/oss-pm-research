# data_analysis/utils.py
import pandas as pd
import numpy as np
import json

def load_data(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

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
    
    return metrics_df, interaction_data, community_df, lifecycle_df, growth_df