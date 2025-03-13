# data_analysis/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_methodology_distribution(data, output_path=None):
    """Create a bar chart showing distribution of project management methodologies."""
    methodologies = [repo.get('pm_classification', {}).get('primary_methodology', 'Unknown') for repo in data]
    methodology_counts = pd.Series(methodologies).value_counts()
    
    plt.figure(figsize=(10, 6))
    methodology_counts.plot(kind='bar')
    plt.title('Distribution of Project Management Methodologies')
    plt.xlabel('Methodology')
    plt.ylabel('Number of Projects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_success_metrics_by_methodology(metrics_df, output_path=None):
    """Create visualizations of success metrics by methodology."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='methodology', y='issue_closure_rate', data=metrics_df)
    plt.title('Issue Closure Rate by Project Management Methodology')
    plt.xlabel('Methodology')
    plt.ylabel('Issue Closure Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_team_interaction_vs_performance(interaction_data, output_path=None):
    """Create scatter plot of team interaction vs project performance."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='team_interaction', y='issue_closure_rate', data=interaction_data)
    plt.title('Team Interaction vs Issue Closure Rate')
    plt.xlabel('Team Interaction Score')
    plt.ylabel('Issue Closure Rate')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_growth_rate_by_methodology(metrics_df, output_path=None):
    """Create visualization of growth rate by methodology."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='methodology', y='growth_rate', data=metrics_df)
    plt.title('Growth Rate by Project Management Methodology')
    plt.xlabel('Methodology')
    plt.ylabel('Growth Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_development_continuity_by_methodology(lifecycle_df, output_path=None):
    lifecycle_cross_tab = pd.crosstab(
        lifecycle_df['methodology'], 
        lifecycle_df['development_stage'],
        values=lifecycle_df['issue_closure_rate'],
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    lifecycle_cross_tab.plot(kind='bar', rot=45)
    plt.title('Issue Closure Rate by Methodology Across Development Stages')
    plt.xlabel('Project Management Methodology')
    plt.ylabel('Average Issue Closure Rate')
    plt.legend(title='Development Stage')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_growth_rate_by_age(growth_df, output_path=None):
    # Calculate growth metrics
    growth_df['stars_per_day'] = growth_df['stars'] / growth_df['age_days']
    growth_df['forks_per_day'] = growth_df['forks'] / growth_df['age_days']

    # Group by methodology
    growth_by_methodology = growth_df.groupby('methodology').agg({
        'star_growth_rate': 'mean',
        'fork_growth_rate': 'mean',
        'stars_per_day': 'mean',
        'forks_per_day': 'mean',
        'age_days': 'mean'
    }).round(2)

    # Visualize
    plt.figure(figsize=(12, 6))
    growth_by_methodology['star_growth_rate'].plot(kind='bar')
    plt.title('Star Growth Rate by Project Management Methodology')
    plt.xlabel('Methodology')
    plt.ylabel('Average Star Growth Rate (stars/day)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()