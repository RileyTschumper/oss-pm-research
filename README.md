# OSS Project Management Approach Research

This repository contains the code for researching how project management approaches influence the long-term success metrics of open-source software projects.

## Overview

This research project aims to analyze GitHub repositories to identify project management methodologies used in open-source software (OSS) development and correlate these approaches with project success metrics. The code is designed to:

1. Collect data from diverse OSS projects on GitHub
2. Analyze communication patterns and project structure
3. Classify project management methodologies
4. Correlate methodologies with project success metrics

## Setup

### Prerequisites

- Python 3.8+
- GitHub Personal Access Token with repo permissions

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your GitHub token as an environment variable:
   ```
   export GITHUB_TOKEN=your_github_token
   ```

## Usage

### Basic Usage

Run the main script to collect and analyze data:

```bash
python main.py
```

### Advanced Options

```bash
python main.py --token YOUR_GITHUB_TOKEN --output data_folder --sample-size 200 --time-window 52
```

For testing different classifier versions on the same data:
```bash
# Reuse an existing repository sample:
python main.py --token YOUR_GITHUB_TOKEN --sample-file data/raw/repo_sample_20240321_123456.json

# Reuse existing processed data (fastest, skips all GitHub API calls):
python main.py --token YOUR_GITHUB_TOKEN --processed-data-file data/processed/all_repositories_20240321_123456.json --action analyze
```

Parameters:
- `--token`: GitHub API token
- `--output`: Output directory for collected data
- `--sample-size`: Number of repositories to sample
- `--time-window`: Number of weeks to analyze for each repository
- `--action`: Choose 'collect', 'analyze', or 'all'
- `--sample-file`: Path to existing repository sample file to reuse instead of collecting new data
- `--processed-data-file`: Path to existing processed data file to reuse instead of reprocessing repositories

## Data Collection

The data collection component (`data_collection/`) uses GitHub's API to gather comprehensive data about repositories, including:

- Basic repository information
- Issues and pull requests
- Project boards and labels
- Communication patterns and timelines
- Contributor activities

Key components:
- `main.py`: Orchestrates the data collection process
- `github_rate_limit_handler.py`: Manages GitHub API requests and rate limits
- `data_collector.py`: Collects repository data with balanced sampling
- `pm_classifier.py`: Classifies project management methodologies

The `GitHubRateLimitHandler` ensures the code stays within GitHub API rate limits, preventing request failures.

## Project Management Classification

The `AdvancedMethodologyClassifier` analyzes repository data to identify project management methodologies, including:

- Agile/Scrum
- Kanban
- XP (Extreme Programming)
- Shape Up
- Waterfall
- ScrumBan
- Ad-hoc/Undefined

Classification is based on multiple feature types:
- Text features from issues, PRs, and comments
- Temporal patterns in contributions and releases
- Structural features of project organization
- Workflow patterns in team interactions

## Data Analysis

The data analysis component (`data_analysis/`) processes and analyzes the collected data to identify patterns and correlations between project management approaches and success metrics. Key components include:

- `analysis.py`: Core analysis functions for processing repository data
- `visualization.py`: Tools for creating visualizations of analysis results
- `statistical_tests.py`: Statistical analysis and hypothesis testing
- `utils.py`: Utility functions for data processing and analysis

The analysis focuses on:
- Issue closure rates and patterns
- Community growth and engagement metrics
- Project longevity and development continuity
- Contributor satisfaction and retention indicators
- Statistical correlations between methodologies and success metrics

## Project Structure

- `data_collection/`: Data collection components
  - `main.py`: Entry point for data collection
  - `github_rate_limit_handler.py`: GitHub API rate limit management
  - `data_collector.py`: Repository data collection
  - `pm_classifier.py`: Project management methodology classification
- `data_analysis/`: Data analysis components
  - `analysis.py`: Core analysis functions
  - `visualization.py`: Data visualization tools
  - `statistical_tests.py`: Statistical analysis
  - `utils.py`: Utility functions
- `requirements.txt`: Lists all required dependencies
- `data/`: Default folder for collected and processed data

## License

[MIT License](LICENSE)

## Acknowledgments

This research project builds upon methodologies discussed in:

