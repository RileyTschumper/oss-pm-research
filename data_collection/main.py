#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for OSS Project Management Research

This script orchestrates the data collection, processing, and analysis
for research on project management approaches in open-source software projects.
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import modules
from github_rate_limit_handler import GitHubRateLimitHandler
from data_collector import BalancedDataCollector
from pm_classifier import AdvancedMethodologyClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("oss_research.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='OSS Project Management Research Data Collection and Analysis'
    )
    parser.add_argument(
        '--token', 
        type=str, 
        default=os.environ.get('GITHUB_TOKEN'),
        help='GitHub API token (default: from GITHUB_TOKEN env variable)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data',
        help='Output directory for collected data (default: "data")'
    )
    parser.add_argument(
        '--sample-file',
        type=str,
        help='Path to existing repository sample file to use instead of collecting new data'
    )
    parser.add_argument(
        '--processed-data-file',
        type=str,
        help='Path to existing processed data file to use instead of reprocessing repositories'
    )
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=50,
        help='Number of repositories to sample (default: 100)'
    )
    parser.add_argument(
        '--time-window', 
        type=int, 
        default=104,
        help='Number of weeks to analyze for each repository (default: 104)'
    )
    parser.add_argument(
        '--action',
        choices=['collect', 'analyze', 'all'],
        default='all',
        help='Action to perform (default: all)'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment for data collection and analysis."""
    # Check GitHub token
    if not args.token:
        raise ValueError("GitHub token is required. Set GITHUB_TOKEN env variable or use --token.")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    paths = {
        'raw': output_dir / 'raw',
        'processed': output_dir / 'processed',
        'analysis': output_dir / 'analysis',
        'results': output_dir / 'results'
    }
    
    for path in paths.values():
        path.mkdir(exist_ok=True)
    
    return paths


def collect_data(github_handler, args, paths):
    """Collect data from GitHub repositories."""
    logger.info("Starting data collection process")
    
    # Initialize timestamp at the start since we'll need it in multiple places
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize the data collector - we'll need it regardless of sample file
    collector = BalancedDataCollector(github_handler)
    
    # Check if using existing processed data file
    if args.processed_data_file:
        processed_path = Path(args.processed_data_file)
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {args.processed_data_file}")
        
        logger.info(f"Loading processed data from {args.processed_data_file}")
        with open(processed_path, 'r') as f:
            processed_data = json.load(f)
        
        if not isinstance(processed_data, list):
            raise ValueError("Invalid processed data file format. Expected a list of repository data.")
            
        logger.info(f"Loaded processed data for {len(processed_data)} repositories")
        return processed_data
    
    # Check if using existing sample file
    if args.sample_file:
        sample_path = Path(args.sample_file)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {args.sample_file}")
        
        logger.info(f"Loading repository sample from {args.sample_file}")
        with open(sample_path, 'r') as f:
            repo_sample = json.load(f)
        
        if not isinstance(repo_sample, list):
            raise ValueError("Invalid sample file format. Expected a list of repositories.")
            
        logger.info(f"Loaded {len(repo_sample)} repositories from sample file")
    else:
        # Sample repositories
        logger.info(f"Sampling {args.sample_size} repositories...")
        repo_sample = collector.get_diverse_repository_sample(args.sample_size)
        
        # Save the raw repository sample
        sample_file = paths['raw'] / f'repo_sample_{timestamp}.json'
        with open(sample_file, 'w') as f:
            json.dump(repo_sample, f, indent=2)
        logger.info(f"Saved repository sample to {sample_file}")
    
    # Process each repository in the sample
    processed_data = []
    for i, repo in enumerate(repo_sample):
        logger.info(f"Processing repository {i+1}/{len(repo_sample)}: {repo['nameWithOwner']}")
        
        try:
            # Get comprehensive repository data
            repo_data = github_handler.get_repository_data(repo['nameWithOwner'])
            
            # Analyze lifecycle indicators
            lifecycle_data = collector.analyze_lifecycle_indicators(repo_data)
            
            # Combine data
            combined_data = {
                'repository': repo['nameWithOwner'],
                'basic_info': repo,
                'detailed_data': repo_data,
                'lifecycle_indicators': lifecycle_data
            }
            
            # Save individual repository data
            repo_file = paths['processed'] / f"{repo['nameWithOwner'].replace('/', '_')}.json"
            with open(repo_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
                
            processed_data.append(combined_data)
            
        except Exception as e:
            logger.error(f"Error processing repository {repo['nameWithOwner']}: {str(e)}")
    
    # Save all processed data
    with open(paths['processed'] / f'all_repositories_{timestamp}.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Data collection completed. Processed {len(processed_data)} repositories.")
    return processed_data


def analyze_data(processed_data, args, paths):
    """Analyze collected data to classify project management approaches."""
    logger.info("Starting data analysis process")
    
    # Initialize the PM classifier
    classifier = AdvancedMethodologyClassifier()
    
    # Process each repository
    results = []
    for i, repo_data in enumerate(processed_data):
        logger.info(f"Analyzing repository {i+1}/{len(processed_data)}: {repo_data['repository']}")
        
        try:
            # Classify project management methodology
            pm_classification = classifier.classify_repository(repo_data['detailed_data'])
            
            # Combine with lifecycle indicators
            analysis_result = {
                'repository': repo_data['repository'],
                'pm_classification': pm_classification,
                'lifecycle_indicators': repo_data['lifecycle_indicators']
            }
            
            # Add basic repository metrics
            analysis_result.update({
                'stars': repo_data['basic_info'].get('stargazerCount', 0),
                'forks': repo_data['basic_info'].get('forkCount', 0),
                'issues': repo_data['basic_info'].get('issues', {}).get('totalCount', 0),
                'pull_requests': repo_data['basic_info'].get('pullRequests', {}).get('totalCount', 0),
                'created_at': repo_data['basic_info'].get('createdAt'),
                'updated_at': repo_data['basic_info'].get('updatedAt'),
            })
            
            results.append(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_data['repository']}: {str(e)}")
    
    # Save analysis results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(paths['analysis'] / f'analysis_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Convert to DataFrame for easier analysis
    df = pd.json_normalize(results)
    df.to_csv(paths['analysis'] / f'analysis_results_{timestamp}.csv', index=False)
    
    logger.info(f"Data analysis completed. Analyzed {len(results)} repositories.")
    return results


def generate_summary(analysis_results, paths):
    """Generate a summary of the analysis results."""
    logger.info("Generating summary of analysis results")
    
    # Create a DataFrame from the results
    df = pd.json_normalize(analysis_results)
    
    # Methodology distribution
    methodology_counts = df['pm_classification.primary_methodology'].value_counts()
    
    # Average metrics by methodology
    metrics_by_methodology = df.groupby('pm_classification.primary_methodology').agg({
        'stars': 'mean',
        'forks': 'mean',
        'issues': 'mean',
        'pull_requests': 'mean',
        'lifecycle_indicators.activity_pattern': lambda x: x.mode().iloc[0] if not x.empty else None,
        'lifecycle_indicators.community_health.contributor_growth': 'mean',
    }).reset_index()
    
    # Success metrics correlation
    metrics_cols = ['stars', 'forks', 'issues', 'pull_requests']
    correlation_matrix = df[metrics_cols].corr()
    
    # Save summary results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as CSV
    methodology_counts.to_csv(paths['results'] / f'methodology_distribution_{timestamp}.csv')
    metrics_by_methodology.to_csv(paths['results'] / f'metrics_by_methodology_{timestamp}.csv', index=False)
    correlation_matrix.to_csv(paths['results'] / f'correlation_matrix_{timestamp}.csv')
    
    # Save as JSON for further processing
    summary = {
        'methodology_distribution': methodology_counts.to_dict(),
        'metrics_by_methodology': metrics_by_methodology.to_dict(orient='records'),
        'correlation_matrix': correlation_matrix.to_dict()
    }
    
    with open(paths['results'] / f'summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary generation completed. Results saved to {paths['results']}")
    return summary


def main():
    """Main execution function."""
    start_time = time.time()
    logger.info("Starting OSS Project Management Research")
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    paths = setup_environment(args)
    
    # Initialize GitHub rate limit handler
    github_handler = GitHubRateLimitHandler(args.token)
    
    # Check rate limit before starting
    rate_limit_data = github_handler.check_rate_limit()
    logger.info(f"Initial GitHub API rate limit: {json.dumps(rate_limit_data['resources'], indent=2)}")
    
    processed_data = []
    analysis_results = []
    
    try:
        # Data collection phase
        if args.action in ['collect', 'all']:
            processed_data = collect_data(github_handler, args, paths)
        elif args.action == 'analyze':
            # Load existing data if only analyzing
            latest_processed = max(paths['processed'].glob('all_repositories_*.json'), 
                                   key=os.path.getctime, default=None)
            if latest_processed:
                logger.info(f"Loading processed data from {latest_processed}")
                with open(latest_processed, 'r') as f:
                    processed_data = json.load(f)
            else:
                logger.error("No processed data found. Run with --action collect first.")
                return
        
        # Data analysis phase
        if args.action in ['analyze', 'all'] and processed_data:
            analysis_results = analyze_data(processed_data, args, paths)
            
            # Generate summary
            generate_summary(analysis_results, paths)
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
    
    finally:
        # Check final rate limit status
        final_rate_limit = github_handler.check_rate_limit()
        logger.info(f"Final GitHub API rate limit: {json.dumps(final_rate_limit['resources'], indent=2)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Execution completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
