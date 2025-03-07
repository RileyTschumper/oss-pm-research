#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Balanced Data Collector for OSS Research

This module provides functionality to collect a balanced and diverse sample
of repositories from GitHub for OSS project management research.
"""

from collections import defaultdict
import logging
import random
import time
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

class BalancedDataCollector:
    """
    Collects a balanced dataset of GitHub repositories for research purposes.
    
    This class provides methods to sample repositories with different characteristics
    to ensure a diverse and representative dataset for research.
    """
    
    def __init__(self, github_client):
        """
        Initialize with a GitHub client.
        
        Args:
            github_client: An instance of GitHubRateLimitHandler
        """
        self.github_client = github_client
        self.logger = logging.getLogger(__name__)
        
    def get_diverse_repository_sample(self, sample_size: int = 100) -> List[Dict[str, Any]]:
        """
        Collect a balanced sample of repositories with diverse characteristics.
        
        Args:
            sample_size: Number of repositories to sample
            
        Returns:
            List of repository data dictionaries
        """
        self.logger.info(f"Collecting a diverse sample of {sample_size} repositories")
        
        """
        # Define queries for different repository categories
        query_categories = {
            'active': 'stars:>100 created:>2020-01-01 language:python,javascript,java,c++,go fork:false archived:false',
            'mature': 'stars:>500 created:2017-01-01..2019-12-31 language:python,javascript,java,c++,go fork:false archived:false',
            'archived': 'archived:true stars:>100 language:python,javascript,java,c++,go',
            'inactive': 'pushed:<2023-01-01 created:<2022-01-01 stars:>100 fork:false',
            'small_team': 'stars:50..200 fork:false archived:false',
            'large_team': 'stars:>1000 fork:false archived:false',
            'educational': 'topic:education fork:false archived:false',
            'tool': 'topic:tool fork:false archived:false',
            'library': 'topic:library fork:false archived:false',
            'framework': 'topic:framework fork:false archived:false'
        }
        """

        # Define queries for different repository categories
        query_categories = {
            'active': 'stars:>100 created:>2020-01-01 language:python,javascript,java,c++,go fork:false archived:false',
            'mature': 'stars:>500 created:2017-01-01..2019-12-31 language:python,javascript,java,c++,go fork:false archived:false',
            'archived': 'archived:true stars:>100 language:python,javascript,java,c++,go',
            'inactive': 'pushed:<2023-01-01 created:<2022-01-01 stars:>100 fork:false'
        }

        # Calculate how many repositories to sample from each category
        category_counts = {}
        base_count = sample_size // len(query_categories)
        remainder = sample_size % len(query_categories)
        
        for category in query_categories:
            category_counts[category] = base_count
            if remainder > 0:
                category_counts[category] += 1
                remainder -= 1
        
        # Collect repositories from each category
        sample_data = []
        
        for category, count in category_counts.items():
            self.logger.info(f"Collecting {count} repositories from category: {category}")
            query = query_categories[category]
            
            # Use GraphQL to search for repositories
            graphql_query = """
            query SearchRepositories($query: String!, $cursor: String) {
                search(query: $query, type: REPOSITORY, first: 100, after: $cursor) {
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                    nodes {
                        ... on Repository {
                            nameWithOwner
                            url
                            createdAt
                            updatedAt
                            pushedAt
                            isArchived
                            stargazerCount
                            forkCount
                            hasProjectsEnabled
                            hasIssuesEnabled
                            description
                            primaryLanguage {
                                name
                            }
                            languages(first: 10) {
                                nodes {
                                    name
                                }
                            }
                            issues(states: OPEN) {
                                totalCount
                            }
                            pullRequests(states: OPEN) {
                                totalCount
                            }
                            defaultBranchRef {
                                name
                                target {
                                    ... on Commit {
                                        history {
                                            totalCount
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
            
            # Initialize pagination
            has_next_page = True
            cursor = None
            category_repos = []
            
            # Fetch repositories until we have enough or there are no more
            while has_next_page and len(category_repos) < count:
                variables = {"query": query, "cursor": cursor}
                
                # Try GraphQL first, fall back to REST API if needed
                try:
                    # Add retry logic for server errors
                    max_retries = 3
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            result = self.github_client.execute_graphql_query(graphql_query, variables)
                            
                            if 'errors' in result:
                                self.logger.error(f"GraphQL query error: {result['errors']}")
                                # Only retry certain types of errors
                                if any("Server error" in str(err) for err in result.get('errors', [])):
                                    retry_count += 1
                                    wait_time = 2 ** retry_count  # exponential backoff
                                    self.logger.info(f"Server error detected, retrying in {wait_time} seconds...")
                                    time.sleep(wait_time)
                                    continue
                                else:
                                    # Non-retriable error
                                    break
                            
                            search_results = result.get('data', {}).get('search', {})
                            page_info = search_results.get('pageInfo', {})
                            has_next_page = page_info.get('hasNextPage', False)
                            cursor = page_info.get('endCursor', None)
                            
                            nodes = search_results.get('nodes', [])
                            category_repos.extend(nodes)
                            
                            # Success, break retry loop
                            break
                            
                        except Exception as e:
                            error_message = str(e)
                            
                            # Check if this is a server error (5xx)
                            if "502 Server Error" in error_message or "503 Service Unavailable" in error_message or "403 Client Error" in error_message:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    self.logger.error(f"Failed after {max_retries} retries: {error_message}")
                                    # Try using the fallback REST API method
                                    break
                                
                                wait_time = 2 ** retry_count  # exponential backoff
                                self.logger.info(f"Server error: {error_message}. Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                # Not a retriable error
                                self.logger.error(f"Error fetching repositories for category {category}: {error_message}")
                                break
                    
                    # If we've exhausted retries or had a non-server error, try REST API
                    if retry_count >= max_retries:
                        # Fall back to REST API search
                        self.logger.info(f"Falling back to REST API for category {category}")
                        rest_repos = self._get_repositories_via_rest(query, 30)  # Get a smaller sample
                        category_repos.extend(rest_repos)
                        has_next_page = False  # Stop after one REST API request
                
                except Exception as e:
                    self.logger.error(f"Unhandled error fetching repositories for category {category}: {str(e)}")
                    # Wait a bit before continuing to the next category
                    time.sleep(5)
                    break
            
            # If we collected more than needed, randomly sample the required count
            if len(category_repos) > count:
                category_repos = random.sample(category_repos, count)
            
            sample_data = self.get_main_repositories(sample_data)
            
            # Add to the final sample
            sample_data.extend(category_repos)
            self.logger.info(f"Collected {len(category_repos)} repositories for category {category}")
        
        self.logger.info(f"Total repositories collected: {len(sample_data)}")
        return sample_data

    def get_main_repositories(self, repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Select the highest-starred repository for each unique project.
            """
            repo_dict = defaultdict(lambda: {"stargazerCount": 0, "repo": None})
            
            for repo in repos:
                project_name = repo["nameWithOwner"].split("/")[0]  # Extract project owner
                if repo["stargazerCount"] > repo_dict[project_name]["stargazerCount"]:
                    repo_dict[project_name] = {"stargazerCount": repo["stargazerCount"], "repo": repo}
            
            return [entry["repo"] for entry in repo_dict.values() if entry["repo"]]

    def _get_repositories_via_rest(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        """
        Use GitHub REST API as a fallback when GraphQL fails.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of repository data dictionaries
        """
        try:
            # Use the search repositories endpoint
            search_url = f"{self.github_client.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results, 100)  # Max 100 per page
            }
            
            response_data = self.github_client.make_request(search_url, params=params)
            repos = response_data.get('items', [])
            
            # Transform REST API response to match GraphQL format
            transformed_repos = []
            for repo in repos:
                transformed = {
                    "nameWithOwner": repo.get('full_name'),
                    "url": repo.get('html_url'),
                    "createdAt": repo.get('created_at'),
                    "updatedAt": repo.get('updated_at'),
                    "pushedAt": repo.get('pushed_at'),
                    "isArchived": repo.get('archived', False),
                    "stargazerCount": repo.get('stargazers_count', 0),
                    "forkCount": repo.get('forks_count', 0),
                    "hasProjectsEnabled": repo.get('has_projects', False),
                    "hasIssuesEnabled": repo.get('has_issues', False),
                    "description": repo.get('description', ''),
                }
                
                # Add basic language info
                if repo.get('language'):
                    transformed["primaryLanguage"] = {"name": repo.get('language')}
                
                # Add basic issue and PR counts if available
                transformed["issues"] = {"totalCount": repo.get('open_issues_count', 0)}
                transformed["pullRequests"] = {"totalCount": 0}  # Not available in basic response
                
                transformed_repos.append(transformed)
            
            return transformed_repos
            
        except Exception as e:
            self.logger.error(f"Error using REST API fallback: {str(e)}")
            return []
    
    def analyze_lifecycle_indicators(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Analyze repository lifecycle stage and health indicators.
        
        Args:
            repo_data: Repository data from GitHubRateLimitHandler.get_repository_data()
            
        Returns:
            Dictionary containing lifecycle indicators
        """
        self.logger.info(f"Analyzing lifecycle indicators for {repo_data.get('info', {}).get('full_name', 'unknown')}")
        
        try:
            return {
                'activity_pattern': self._classify_activity_pattern(repo_data),
                'growth_trajectory': self._analyze_growth_trajectory(repo_data),
                'community_health': self._assess_community_health(repo_data),
                'development_continuity': self._measure_development_continuity(repo_data)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing lifecycle indicators: {str(e)}")
            return {
                'activity_pattern': 'unknown',
                'growth_trajectory': {},
                'community_health': {},
                'development_continuity': {}
            }
    
    def _classify_activity_pattern(self, repo_data: Dict) -> str:
        """
        Classify the activity pattern of a repository.
        
        Args:
            repo_data: Repository data
            
        Returns:
            Activity pattern classification
        """
        # Get commit activity data
        commit_activity = repo_data.get('statistics', {}).get('commit_activity', [])
        
        if not commit_activity:
            return 'unknown'
        
        # Extract weekly commit counts
        weekly_commits = [week.get('total', 0) for week in commit_activity]
        
        if not weekly_commits:
            return 'unknown'
        
        # Calculate metrics
        avg_commits = np.mean(weekly_commits)
        std_commits = np.std(weekly_commits)
        zero_weeks = sum(1 for count in weekly_commits if count == 0)
        
        # Classify based on metrics
        if avg_commits < 1:
            return 'inactive'
        elif zero_weeks > len(weekly_commits) / 2:
            return 'sporadic'
        elif std_commits / max(avg_commits, 1) > 1.5:
            return 'burst'
        elif avg_commits > 10:
            return 'very_active'
        else:
            return 'steady'

    def _analyze_growth_trajectory(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Analyze the growth pattern of the repository.
        
        Args:
            repo_data: Repository data
            
        Returns:
            Growth trajectory indicators
        """
        # Get stargazer count and creation date
        info = repo_data.get('info', {})
        stars = info.get('stargazers_count', 0)
        created_at = info.get('created_at', '')
        if not created_at:
            return {
                'trend': 'unknown',
                'star_growth_rate': 0,
                'fork_growth_rate': 0
            }
        
        # Calculate age in days - Make both datetime objects timezone aware
        try:
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            # Make sure current time is also timezone aware
            current_time = datetime.now().astimezone()
            age_days = (current_time - created_date).days
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error parsing dates: {str(e)}")
            return {
                'trend': 'unknown',
                'star_growth_rate': 0,
                'fork_growth_rate': 0
            }
        
        if age_days <= 0:
            return {
                'trend': 'unknown',
                'star_growth_rate': 0,
                'fork_growth_rate': 0
            }
        
        # Calculate growth rates
        star_growth_rate = stars / age_days
        fork_growth_rate = info.get('forks_count', 0) / age_days
        
        # Classify growth trend
        if star_growth_rate > 1:  # More than 1 star per day
            trend = 'explosive'
        elif star_growth_rate > 0.1:  # More than 1 star per 10 days
            trend = 'strong'
        elif star_growth_rate > 0.01:  # More than 1 star per 100 days
            trend = 'moderate'
        else:
            trend = 'slow'
        
        return {
            'trend': trend,
            'star_growth_rate': star_growth_rate,
            'fork_growth_rate': fork_growth_rate
        }

    def _assess_community_health(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Assess various aspects of community health.
        
        Args:
            repo_data: Repository data
            
        Returns:
            Community health indicators
        """
        # Extract relevant data
        info = repo_data.get('info', {})
        issues = repo_data.get('issues', [])
        pulls = repo_data.get('pull_requests', [])
        
        # Calculate metrics
        total_issues = len(issues)
        open_issues = sum(1 for issue in issues if issue.get('state') == 'open')
        issue_closure_rate = 1 - (open_issues / max(total_issues, 1))
        
        total_pulls = len(pulls)
        open_pulls = sum(1 for pr in pulls if pr.get('state') == 'open')
        pr_acceptance_rate = 1 - (open_pulls / max(total_pulls, 1))
        
        # Calculate response times for issues
        issue_response_times = []
        for issue in issues:
            created_at = issue.get('created_at', '')
            comments = issue.get('comments', 0)
            
            if created_at and comments > 0 and 'updated_at' in issue:
                try:
                    created_at_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    updated_at_dt = datetime.fromisoformat(issue.get('updated_at', '').replace('Z', '+00:00'))
                    response_time = (updated_at_dt - created_at_dt).total_seconds() / 3600  # hours
                    issue_response_times.append(response_time)
                except (ValueError, TypeError) as e:
                    # Skip issues with malformed dates
                    self.logger.debug(f"Skipping issue with malformed dates: {str(e)}")
                    continue
        
        avg_response_time = np.mean(issue_response_times) if issue_response_times else 0
        
        # Estimate number of active contributors
        issue_creators = set()
        pr_creators = set()
        
        for issue in issues:
            if issue.get('user', {}).get('login'):
                issue_creators.add(issue.get('user', {}).get('login'))
                
        for pr in pulls:
            if pr.get('user', {}).get('login'):
                pr_creators.add(pr.get('user', {}).get('login'))
                
        unique_contributors = len(issue_creators.union(pr_creators))
        
        # Calculate contributor growth (simple estimate)
        contributors = list(issue_creators.union(pr_creators))
        
        # Safely calculate contributor growth
        created_at = info.get('created_at', '')
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                # Ensure current time is timezone aware
                current_time = datetime.now().astimezone()
                age_days = max((current_time - created_date).days, 1)
                contributor_growth = len(contributors) / age_days * 30  # per month
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error calculating contributor growth: {str(e)}")
                contributor_growth = 0
        else:
            contributor_growth = 0
        
        return {
            'issue_closure_rate': issue_closure_rate,
            'pr_acceptance_rate': pr_acceptance_rate,
            'avg_issue_response_time': avg_response_time,
            'unique_contributors': unique_contributors,
            'contributor_growth': contributor_growth
        }

    
    def _measure_development_continuity(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Measure the continuity of development activities.
        
        Args:
            repo_data: Repository data
            
        Returns:
            Development continuity indicators
        """
        # Extract commit history
        commit_activity = repo_data.get('statistics', {}).get('commit_activity', [])
        
        if not commit_activity:
            return {
                'continuity_score': 0,
                'development_stage': 'unknown',
                'max_inactive_period': 0
            }
        
        # Calculate the longest period without commits (in weeks)
        weekly_commits = [week.get('total', 0) for week in commit_activity]
        max_inactive_streak = 0
        current_streak = 0
        
        for count in weekly_commits:
            if count == 0:
                current_streak += 1
            else:
                max_inactive_streak = max(max_inactive_streak, current_streak)
                current_streak = 0
        
        max_inactive_streak = max(max_inactive_streak, current_streak)
        
        # Calculate continuity score (higher is better)
        active_weeks = sum(1 for count in weekly_commits if count > 0)
        continuity_score = active_weeks / max(len(weekly_commits), 1)
        
        # Determine development stage
        info = repo_data.get('info', {})
        created_at = info.get('created_at', '')
        
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                # Make current time timezone aware
                current_time = datetime.now().astimezone()
                age_days = (current_time - created_date).days
                
                if age_days < 90:  # Less than 3 months
                    development_stage = 'initial'
                elif age_days < 365:  # Less than 1 year
                    development_stage = 'growth'
                elif age_days < 1095:  # Less than 3 years
                    development_stage = 'mature'
                else:
                    development_stage = 'established'
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error determining development stage: {str(e)}")
                development_stage = 'unknown'
        else:
            development_stage = 'unknown'
        
        return {
            'continuity_score': continuity_score,
            'development_stage': development_stage,
            'max_inactive_period': max_inactive_streak
        }
