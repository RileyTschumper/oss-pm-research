#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GitHub Rate Limit Handler

This module provides a class to handle GitHub API rate limits efficiently,
avoiding hitting rate limits by waiting when necessary and managing requests.
"""

import re
import requests
import random
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

class GitHubRateLimitHandler:
    """
    Handles GitHub API requests with rate limit awareness.
    
    This class makes GitHub API requests while being aware of rate limits,
    waiting when necessary to avoid hitting limits.
    """
    
    def __init__(self, token: str):
        """
        Initialize with a GitHub token.
        
        Args:
            token: GitHub personal access token
        """
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_rate_limit(self) -> Dict[str, Any]:
        """
        Check current rate limit status.
        
        Returns:
            Dict containing rate limit information
        """
        response = requests.get(f'{self.base_url}/rate_limit', headers=self.headers)
        return response.json()

    def wait_for_reset_if_needed(self, rate_limit_data: Dict[str, Any], buffer: int = 1) -> None:
        """
        Wait if we're close to rate limit.
        
        Args:
            rate_limit_data: Rate limit data from check_rate_limit()
            buffer: Number of requests to keep as buffer
        """
        for api_type in ['core', 'graphql', 'search']:
            limit_data = rate_limit_data['resources'][api_type]
            remaining = limit_data['remaining']
            reset_time = limit_data['reset']

            if remaining < buffer:  # Buffer to avoid hitting the limit
                wait_time = reset_time - int(time.time()) + 60  # Add 60 second buffer
                if wait_time > 0:
                    # self.logger.info(f"Rate limit for {api_type} approaching. Waiting {wait_time} seconds...")
                    self.logger.info(f"Rate limit for {api_type} approaching. Limit: {limit_data['limit']}, Used: {limit_data['limit'] - limit_data['remaining']}, Remaining: {limit_data['remaining']}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)

    def make_request(self, 
                     url: str, 
                     method: str = 'get', 
                     params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict:
        """
        Make a rate-limit aware request.
        
        Args:
            url: URL to request
            method: HTTP method (get or post)
            params: Query parameters for GET
            data: JSON data for POST
            
        Returns:
            JSON response as dictionary
        """
        rate_limit_data = self.check_rate_limit()
        self.wait_for_reset_if_needed(rate_limit_data)

        if method.lower() == 'post':
            response = requests.post(url, headers=self.headers, json=data)
        else:
            response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            self.logger.warning("Rate limit exceeded during request. Waiting for reset...")
            rate_limit_data = self.check_rate_limit()
            self.wait_for_reset_if_needed(rate_limit_data)
            return self.make_request(url, method, params, data)  # Retry request

        response.raise_for_status()
        return response.json()

    def get_paginated_data(self, url: str, params: Optional[Dict] = None, sample_size: int = 10) -> List[Dict]:
        """
        Fetch a random sample of pages instead of sequential pagination.

        Args:
            url: URL to request
            params: Query parameters
            sample_size: Number of random pages to fetch (default is 10)
        
        Returns:
            List of all items from the sampled pages
        """
        if params is None:
            params = {}

        params['page'] = 1
        params['per_page'] = 100
        response = self.make_request(url, params=params)

        if not response:
            return []

        first_page_data = list(response)
        all_data = first_page_data

        # Extract 'Link' header using a separate request with headers
        headers = self.get_response_headers(url, params=params)  # New method to get headers

        # Extract last page number from the 'Link' header
        last_page = 1
        link_header = headers.get('Link', '')
        match = re.search(r'page=(\d+)>; rel="last"', link_header)
        if match:
            last_page = int(match.group(1))
            self.logger.info(f"last page {last_page}")

        if last_page <= 1:
            return all_data  # No pagination needed

        # Randomly select page numbers (excluding the first page we already fetched)
        sampled_pages = random.sample(range(2, last_page + 1), min(sample_size, last_page - 1))


        """
        # Fetch the first page to determine total count
        params['page'] = 1
        params['per_page'] = 100
        first_page_data = self.make_request(url, params=params)

        if not first_page_data:
            return []

        all_data = list(first_page_data)

        # Determine total number of pages
        total_items = len(first_page_data)
        if total_items < 100:
            return all_data  # No need for pagination

        # Estimate total pages (GitHub does not always return this directly)
        total_count_response = self.make_request(url, params={**params, 'per_page': 1})
        total_count = total_count_response[0]['total_count'] if 'total_count' in total_count_response[0] else (len(first_page_data) * 10)
        total_pages = min(10, (total_count // 100) + (1 if total_count % 100 > 0 else 0))

        if total_pages <= 1:
            return all_data

        # Randomly select page numbers (excluding the first page we already fetched)
        sampled_pages = random.sample(range(2, total_pages + 1), min(sample_size, total_pages - 1))
        """


        for page in sampled_pages:
            params['page'] = page
            sampled_data = self.make_request(url, params=params)
            if sampled_data:
                all_data.extend(sampled_data)

            self.logger.info(f"Fetched random page {page} from {url}")

        return all_data

    def get_response_headers(self, url: str, params: Optional[Dict] = None) -> Dict[str, str]:
        """
        Fetch only the headers of a request.

        Args:
            url: API URL
            params: Query parameters

        Returns:
            Dictionary of response headers
        """
        try:
            # Check rate limits before making the request
            rate_limit_data = self.check_rate_limit()
            self.wait_for_reset_if_needed(rate_limit_data)
            
            # Make authenticated request with proper headers
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                self.logger.warning("Rate limit exceeded during header request. Waiting for reset...")
                rate_limit_data = self.check_rate_limit()
                self.wait_for_reset_if_needed(rate_limit_data)
                # Retry the request after waiting
                response = requests.get(url, headers=self.headers, params=params)
            
            response.raise_for_status()
            return response.headers
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch headers from {url}: {e}")
            return {}

    def get_repository_data(self, repo_full_name: str) -> Dict[str, Any]:
        """
        Get comprehensive repository data with rate limit handling.
        
        Args:
            repo_full_name: Repository name in owner/repo format
            
        Returns:
            Dictionary with repository data
        """
        repo_data = {}
        
        # Basic repository information
        repo_url = f'{self.base_url}/repos/{repo_full_name}'
        repo_data['info'] = self.make_request(repo_url)
        
        """
        # Get project boards
        try:
            projects_url = f'{repo_url}/projects'
            self.logger.info(f'{repo_url}/projects')
            repo_data['projects'] = self.get_paginated_data(projects_url)
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"Could not fetch projects for {repo_full_name}: {e}")
            repo_data['projects'] = []
        """
        
        # Get issues with pagination
        issues_url = f'{repo_url}/issues'
        params = {
            'state': 'all',
            'sort': 'updated',
            'direction': 'desc'
        }
        repo_data['issues'] = self.get_paginated_data(issues_url, params)

        # Get pull requests
        pulls_url = f'{repo_url}/pulls'
        params = {
            'state': 'all',
            'sort': 'updated',
            'direction': 'desc'
        }
        repo_data['pull_requests'] = self.get_paginated_data(pulls_url, params)

        # Get repository statistics
        stats_endpoints = {
            'commit_activity': f'{repo_url}/stats/commit_activity',
            'code_frequency': f'{repo_url}/stats/code_frequency',
            'participation': f'{repo_url}/stats/participation',
            'contributors': f'{repo_url}/stats/contributors'
        }
        
        repo_data['statistics'] = {}
        for stat_name, stat_url in stats_endpoints.items():
            try:
                repo_data['statistics'][stat_name] = self.make_request(stat_url)
            except requests.exceptions.HTTPError as e:
                self.logger.warning(f"Could not fetch {stat_name} for {repo_full_name}: {e}")
                repo_data['statistics'][stat_name] = None
                
        # Get commit history
        commits_url = f'{repo_url}/commits'
        try:
            repo_data['commits'] = self.get_paginated_data(commits_url)
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"Could not fetch commits for {repo_full_name}: {e}")
            repo_data['commits'] = []
            
        # Get issue comments
        try:
            issue_comments_url = f'{repo_url}/issues/comments'
            repo_data['issue_comments'] = self.get_paginated_data(issue_comments_url)
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"Could not fetch issue comments for {repo_full_name}: {e}")
            repo_data['issue_comments'] = []
            
        # Get pull request comments
        try:
            pr_comments_url = f'{repo_url}/pulls/comments'
            repo_data['pull_request_comments'] = self.get_paginated_data(pr_comments_url)
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"Could not fetch PR comments for {repo_full_name}: {e}")
            repo_data['pull_request_comments'] = []

        return repo_data

    def execute_graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a GitHub GraphQL API query.
        
        Args:
            query: GraphQL query string
            variables: Variables for the query
            
        Returns:
            Query response as dictionary
        """
        url = "https://api.github.com/graphql"
        payload = {"query": query}
        
        if variables:
            payload["variables"] = variables
            
        rate_limit_data = self.check_rate_limit()
        self.wait_for_reset_if_needed(rate_limit_data)
        
        response = requests.post(url, json=payload, headers=self.headers)
        
        if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            self.logger.warning("GraphQL rate limit exceeded. Waiting for reset...")
            rate_limit_data = self.check_rate_limit()
            self.wait_for_reset_if_needed(rate_limit_data)
            return self.execute_graphql_query(query, variables)  # Retry query
            
        response.raise_for_status()
        return response.json()

    def get_repository_communication_network(self, repo_full_name: str, time_window: int = 104) -> Dict[str, Any]:
        """
        Get repository communication network data for time window.
        
        Args:
            repo_full_name: Repository name in owner/repo format
            time_window: Number of weeks to analyze
            
        Returns:
            Dictionary with weekly communication network data
        """
        # Get repository creation date
        repo_url = f'{self.base_url}/repos/{repo_full_name}'
        repo_info = self.make_request(repo_url)
        
        created_at = datetime.fromisoformat(repo_info['created_at'].replace('Z', '+00:00'))
        network_data = {}
        
        # Calculate weekly boundaries
        current_time = datetime.now()
        weeks = min(time_window, int((current_time - created_at).days / 7))
        
        self.logger.info(f"Analyzing {weeks} weeks of communication data for {repo_full_name}")
        
        # Collect issue comments by week
        issues_url = f'{repo_url}/issues'
        params = {
            'state': 'all',
            'sort': 'created',
            'direction': 'asc',
            'per_page': 100
        }
        
        issues = self.get_paginated_data(issues_url, params)
        
        # Process each issue and its comments to build weekly networks
        for issue in issues:
            issue_number = issue['number']
            issue_created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
            
            # Calculate week number from repo creation
            week_num = int((issue_created - created_at).days / 7)
            
            if week_num >= weeks:
                continue
                
            if week_num not in network_data:
                network_data[week_num] = {
                    'nodes': set(),
                    'edges': {},
                    'issues_opened': 0,
                    'issues_closed': 0,
                    'pr_opened': 0,
                    'pr_closed': 0
                }
            
            # Record issue activity
            network_data[week_num]['nodes'].add(issue['user']['login'])
            network_data[week_num]['issues_opened'] += 1
            
            # Get issue comments
            comments_url = f'{repo_url}/issues/{issue_number}/comments'
            try:
                comments = self.get_paginated_data(comments_url)
                
                # Process comments to build interaction network
                for comment in comments:
                    comment_created = datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00'))
                    comment_week = int((comment_created - created_at).days / 7)
                    
                    if comment_week >= weeks:
                        continue
                        
                    if comment_week not in network_data:
                        network_data[comment_week] = {
                            'nodes': set(),
                            'edges': {},
                            'issues_opened': 0,
                            'issues_closed': 0,
                            'pr_opened': 0,
                            'pr_closed': 0
                        }
                    
                    commenter = comment['user']['login']
                    network_data[comment_week]['nodes'].add(commenter)
                    
                    # Create edge between issue creator and commenter
                    edge = (issue['user']['login'], commenter)
                    if edge not in network_data[comment_week]['edges']:
                        network_data[comment_week]['edges'][edge] = 0
                    
                    network_data[comment_week]['edges'][edge] += 1
                    
            except requests.exceptions.HTTPError as e:
                self.logger.warning(f"Could not fetch comments for issue #{issue_number}: {e}")
        
        # Convert sets to lists for JSON serialization
        for week_num in network_data:
            network_data[week_num]['nodes'] = list(network_data[week_num]['nodes'])
            network_data[week_num]['edges'] = {f"{k[0]}|{k[1]}": v for k, v in network_data[week_num]['edges'].items()}
        
        return network_data
