# data_analysis/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import re
import glob
from datetime import datetime

from visualization import (
    plot_methodology_distribution,
    plot_success_metrics_by_methodology,
    plot_team_interaction_vs_performance,
    plot_growth_rate_by_methodology,
    plot_development_continuity_by_methodology,
    plot_growth_rate_by_age
)
from statistical_tests import (
    run_methodology_comparison_tests,
    correlation_analysis,
    regression_analysis
)
from utils import load_data, preprocess_data

def load_and_merge_data(file_paths):
    """
    Load multiple JSON files and merge them into a single list
    
    Args:
        file_paths (list): List of file paths to JSON files
        
    Returns:
        list: Combined list of data from all files
    """
    combined_data = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    print(f"Warning: {file_path} does not contain a list of repositories. Skipping.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Remove duplicate repositories based on repository name
    unique_repos = {}
    for repo in combined_data:
        repo_name = repo.get('repository')
        if repo_name and repo_name not in unique_repos:
            unique_repos[repo_name] = repo
    
    return list(unique_repos.values())

def main():
    # Define the file paths for the JSON files
    file_paths = [
        Path('../data/analysis/analysis_results_20250313_215608.json'),
        Path('../data/analysis/analysis_results_20250305_181708.json'),
        Path('../data/analysis/analysis_results_20250304_194315.json')
    ]
    
    # Load and merge data
    data = load_and_merge_data(file_paths)
    
    # Generate timestamp for output files
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Preprocess data
    metrics_df, interaction_data, community_df, lifecycle_df, growth_df = preprocess_data(data)
    
    # Create output directory
    output_dir = Path('../results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses with current timestamp in filenames
    plot_methodology_distribution(data, output_dir / f'methodology_distribution_combined_{current_timestamp}.png')
    plot_success_metrics_by_methodology(metrics_df, output_dir / f'success_metrics_combined_{current_timestamp}.png')
    plot_growth_rate_by_methodology(metrics_df, output_dir / f'growth_rate_combined_{current_timestamp}.png')
    plot_development_continuity_by_methodology(lifecycle_df, output_dir / f'development_continuity_combined_{current_timestamp}.png')
    plot_growth_rate_by_age(growth_df, output_dir / f'growth_rate_by_age_combined_{current_timestamp}.png')

    # Run statistical tests
    test_results = run_methodology_comparison_tests(metrics_df)
    
    # Save test results to a file
    results_file = output_dir / f'statistical_test_results_combined_{current_timestamp}.txt'
    with open(results_file, 'w') as f:
        f.write("Statistical Test Results (Combined Data):\n")
        f.write(f"Number of repositories analyzed: {len(data)}\n\n")
        f.write(str(test_results))
    
    # Print summary
    print(f"Analysis complete. Combined {len(data)} repositories from {len(file_paths)} files.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()