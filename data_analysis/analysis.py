# data_analysis/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import re

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

def main():
    # Load data
    data_path = Path('../data/analysis/analysis_results_20250305_181708.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract date and time from input filename
    timestamp_match = re.search(r'analysis_results_(\d{8}_\d{6})', data_path.name)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        timestamp = "unknown"
    
    # Preprocess data
    metrics_df, interaction_data, community_df, lifecycle_df, growth_df = preprocess_data(data)
    
    # Create output directory
    output_dir = Path('../results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses with timestamp in filenames
    plot_methodology_distribution(data, output_dir / f'methodology_distribution_{timestamp}.png')
    plot_success_metrics_by_methodology(metrics_df, output_dir / f'success_metrics_{timestamp}.png')
    plot_growth_rate_by_methodology(metrics_df, output_dir / f'growth_rate_{timestamp}.png')
    plot_development_continuity_by_methodology(lifecycle_df, output_dir / f'development_continuity_{timestamp}.png')
    plot_growth_rate_by_age(growth_df, output_dir / f'growth_rate_by_age_{timestamp}.png')

    # Run statistical tests
    test_results = run_methodology_comparison_tests(metrics_df)
    
    # Save test results to a file
    results_file = output_dir / f'statistical_test_results_{timestamp}.txt'
    with open(results_file, 'w') as f:
        f.write("Statistical Test Results:\n")
        f.write(str(test_results))
    
    # Print completion message
    print("Analysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()