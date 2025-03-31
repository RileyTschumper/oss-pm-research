#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Project Management Methodology Classifier

This module provides functionality to classify OSS projects according to their
project management methodologies based on their communication patterns,
project structure, and workflow characteristics.
"""

import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class AdvancedMethodologyClassifier:
    """
    Classifies project management methodologies used in OSS projects.
    
    This class analyzes repository data to determine the project management 
    methodology most likely being used by the project team.
    """
    
    def __init__(self):
        """Initialize the classifier with pattern dictionaries and feature extractors."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize text vectorizer for NLP
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Methodology-specific pattern dictionaries
        self.methodology_patterns = {
            'agile_scrum': {
                'terms': ['sprint', 'backlog', 'story point', 'scrum', 'user story'],
                'time_patterns': [(7, 21)],  # Sprint lengths in days (min, max)
                'board_structure': ['backlog', 'sprint', 'in progress', 'review', 'done'],
                'ceremonies': ['daily standup', 'sprint planning', 'retrospective'],
                'doc_patterns': ['user story', 'acceptance criteria']
            },
            'kanban': {
                'terms': ['wip limit', 'pull system', 'flow', 'continuous'],
                'board_structure': ['to do', 'in progress', 'done'],
                'metrics': ['lead time', 'cycle time', 'throughput'],
                'principles': ['visualize work', 'limit wip', 'manage flow']
            },
            'xp': {
                'terms': ['pair programming', 'tdd', 'continuous integration'],
                'practices': ['refactoring', 'unit testing', 'collective ownership'],
                'time_patterns': [(1, 3)],  # Very short iterations
                'metrics': ['test coverage', 'build status']
            },
            'shape_up': {
                'terms': ['appetite', 'betting table', 'pitch'],
                'time_patterns': [(42, 44)],  # 6-week cycles
                'phases': ['shaping', 'betting', 'building'],
                'cool_down': ['cool-down', 'cooldown', 'cool down']
            },
            'waterfall': {
                'terms': ['phase', 'milestone', 'deliverable', 'gantt'],
                'phases': ['requirements', 'design', 'implementation', 'verification', 'maintenance'],
                'doc_patterns': ['requirements document', 'design specification']
            },
            'scrumban': {
                'terms': ['wip limit', 'kanban', 'scrum', 'backlog', 'sprint'],
                'board_structure': ['backlog', 'ready', 'in progress', 'done'],
                'principles': ['pull system', 'visualize workflow', 'limit wip']
            }
        }
        
        # Initialize feature extractors
        self.feature_extractors = [
            self.extract_text_features,
            self.extract_temporal_features,
            self.extract_structural_features,
            self.extract_workflow_features
        ]

    def extract_text_features(self, repo_data: Dict) -> np.ndarray:
        """
        Extract features from text content using NLP.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Array of text features
        """
        # Combine relevant text from issues, PRs, and documentation
        text_content = []
        
        # Process issue titles and descriptions with null checks
        for issue in repo_data.get('issues', []):
            if issue.get('title'):
                text_content.append(issue.get('title', ''))
            if issue.get('body'):
                text_content.append(issue.get('body', ''))
        
        # Process PR descriptions with null checks
        for pr in repo_data.get('pull_requests', []):
            if pr.get('title'):
                text_content.append(pr.get('title', ''))
            if pr.get('body'):
                text_content.append(pr.get('body', ''))
        
        # Process issue comments with null checks
        for comment in repo_data.get('issue_comments', []):
            if comment.get('body'):
                text_content.append(comment.get('body', ''))
        
        # Process PR comments with null checks
        for comment in repo_data.get('pull_request_comments', []):
            if comment.get('body'):
                text_content.append(comment.get('body', ''))
        
        # Process repository description and README
        info = repo_data.get('info', {})
        if info.get('description'):
            text_content.append(info.get('description', ''))
        
        # Filter out any None values and combine text
        text_content = [text for text in text_content if text]
        combined_text = ' '.join(text_content)
        
        if not combined_text or len(combined_text) < 10:
            self.logger.warning("Insufficient text content for analysis")
            return np.zeros((1, 10))  # Return empty feature vector
        
        try:
            # Create feature vector using TF-IDF
            features = self.vectorizer.fit_transform([combined_text])
            
            # If features are too sparse, reduce dimensionality
            if features.shape[1] > 100:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=min(100, features.shape[1]-1))
                features = svd.fit_transform(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting text features: {str(e)}")
            return np.zeros((1, 10))  # Return empty feature vector
        
    def extract_temporal_features(self, repo_data: Dict) -> np.ndarray:
        """
        Analyze temporal patterns in issues and releases.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Array of temporal features
        """
        features = []
        
        try:
            # Analyze issue creation and closure patterns
            issue_dates = []
            for issue in repo_data.get('issues', []):
                created_at = issue.get('created_at')
                if created_at:
                    issue_dates.append(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
            
            if issue_dates:
                # Calculate average time between issues
                sorted_dates = sorted(issue_dates)
                if len(sorted_dates) > 1:
                    intervals = [(sorted_dates[i+1] - sorted_dates[i]).total_seconds() / 86400  # Convert to days
                                for i in range(len(sorted_dates)-1)]
                    
                    features.extend([
                        np.mean(intervals),
                        np.std(intervals),
                        np.median(intervals)
                    ])
                else:
                    features.extend([0, 0, 0])  # No intervals available
                
                # Detect cyclical patterns (potential sprints)
                if len(sorted_dates) > 5:
                    # Group issues by week
                    week_counts = defaultdict(int)
                    for date in sorted_dates:
                        week_key = date.isocalendar()[1]  # Week number
                        week_counts[week_key] += 1
                    
                    # Look for patterns in weekly counts
                    week_values = list(week_counts.values())
                    if len(week_values) > 2:
                        week_mean = np.mean(week_values)
                        week_std = np.std(week_values)
                        features.append(week_std / max(week_mean, 1))  # Coefficient of variation
                    else:
                        features.append(0)
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0, 0])  # No issue dates
            
            # Analyze merge patterns for PRs
            merge_intervals = []
            for pr in repo_data.get('pull_requests', []):
                created_at = pr.get('created_at')
                merged_at = pr.get('merged_at')
                
                if created_at and merged_at:
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    merged_date = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                    merge_intervals.append((merged_date - created_date).total_seconds() / 3600)  # Hours
            
            if merge_intervals:
                features.extend([
                    np.mean(merge_intervals),
                    np.median(merge_intervals),
                    np.std(merge_intervals)
                ])
            else:
                features.extend([0, 0, 0])
            
            # Analyze commit patterns
            commit_stats = repo_data.get('statistics', {}).get('commit_activity', [])
            if commit_stats:
                weekly_commits = [week.get('total', 0) for week in commit_stats]
                
                features.extend([
                    np.mean(weekly_commits),
                    np.std(weekly_commits),
                    np.max(weekly_commits),
                    np.sum(1 for c in weekly_commits if c == 0) / max(len(weekly_commits), 1)  # Zero commit ratio
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            return np.zeros((1, 10))  # Return empty feature vector

    def extract_structural_features(self, repo_data: Dict) -> np.ndarray:
        """
        Analyze project structure and organization.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Array of structural features
        """
        features = []
        
        try:
            # Analyze issue labels
            labels = []
            for issue in repo_data.get('issues', []):
                for label in issue.get('labels', []):
                    label_name = label.get('name', '').lower()
                    if label_name:
                        labels.append(label_name)
            
            # Count methodology-specific labels
            for methodology, patterns in self.methodology_patterns.items():
                methodology_terms = patterns.get('terms', [])
                term_count = sum(1 for label in labels if any(term in label for term in methodology_terms))
                features.append(term_count)
            
            # Analyze project structure indicators
            info = repo_data.get('info', {})
            has_wiki = info.get('has_wiki', False)
            has_projects = info.get('has_projects', False)
            has_issues = info.get('has_issues', False)
            
            features.extend([
                1 if has_wiki else 0,
                1 if has_projects else 0,
                1 if has_issues else 0
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting structural features: {str(e)}")
            return np.zeros((1, 10))  # Return empty feature vector

    def extract_workflow_features(self, repo_data: Dict) -> np.ndarray:
        """
        Analyze workflow patterns and processes.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Array of workflow features
        """
        features = []
        
        try:
            # Analyze PR review patterns
            pr_reviews_count = []
            for pr in repo_data.get('pull_requests', []):
                if pr.get('review_comments', 0) > 0:
                    pr_reviews_count.append(pr.get('review_comments'))
            
            if pr_reviews_count:
                features.extend([
                    np.mean(pr_reviews_count),
                    np.median(pr_reviews_count),
                    np.std(pr_reviews_count) if len(pr_reviews_count) > 1 else 0
                ])
            else:
                features.extend([0, 0, 0])
            
            # Analyze issue closure patterns
            open_issues = sum(1 for issue in repo_data.get('issues', []) if issue.get('state') == 'open')
            closed_issues = sum(1 for issue in repo_data.get('issues', []) if issue.get('state') == 'closed')
            total_issues = open_issues + closed_issues
            
            features.extend([
                closed_issues / max(total_issues, 1),  # Issue closure rate
                total_issues
            ])
            
            # Analyze PR acceptance patterns
            open_prs = sum(1 for pr in repo_data.get('pull_requests', []) if pr.get('state') == 'open')
            merged_prs = sum(1 for pr in repo_data.get('pull_requests', []) if pr.get('merged_at') is not None)
            closed_prs = sum(1 for pr in repo_data.get('pull_requests', []) if pr.get('state') == 'closed' and pr.get('merged_at') is None)
            total_prs = open_prs + merged_prs + closed_prs
            
            features.extend([
                merged_prs / max(total_prs, 1),  # PR merge rate
                closed_prs / max(total_prs, 1),  # PR rejection rate
                total_prs
            ])
            
            # Analyze automation indicators
            has_ci = False
            has_automated_tests = False
            
            # Look for CI/CD indicators in issue/PR comments
            ci_terms = ['ci', 'cd', 'continuous integration', 'continuous delivery', 
                    'travis', 'jenkins', 'github actions', 'circleci']
            test_terms = ['test', 'coverage', 'automated checks', 'build passed']
            
            # Add proper null checks for comment bodies
            for comment in repo_data.get('issue_comments', []) + repo_data.get('pull_request_comments', []):
                body = comment.get('body', '').lower() if comment.get('body') else ''
                if any(term in body for term in ci_terms):
                    has_ci = True
                if any(term in body for term in test_terms):
                    has_automated_tests = True
            
            features.extend([
                1 if has_ci else 0,
                1 if has_automated_tests else 0
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting workflow features: {str(e)}")
            return np.zeros((1, 10))  # Return empty feature vector

    def _calculate_sequence_similarity(self, sequence1: List[str], sequence2: List[str]) -> float:
        """
        Calculate similarity between two sequences.
        
        Args:
            sequence1: First sequence of strings
            sequence2: Second sequence of strings
            
        Returns:
            Similarity score between 0 and 1
        """
        if not sequence1 or not sequence2:
            return 0.0
            
        # Normalize sequences
        seq1 = [s.lower().strip() for s in sequence1]
        seq2 = [s.lower().strip() for s in sequence2]
        
        # Calculate jaccard similarity
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / max(union, 1)

    def _extract_all_features(self, repo_data: Dict) -> np.ndarray:
        """
        Extract all features using multiple extractors.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Combined feature array
        """
        feature_sets = []
        
        for extractor in self.feature_extractors:
            try:
                features = extractor(repo_data)
                feature_sets.append(features)
            except Exception as e:
                self.logger.error(f"Error in feature extractor {extractor.__name__}: {str(e)}")
                # Add empty features to maintain structure
                feature_sets.append(np.zeros((1, 1)))
        
        # Combine features
        try:
            combined_features = np.concatenate(feature_sets, axis=1)
            return combined_features
        except Exception as e:
            self.logger.error(f"Error combining features: {str(e)}")
            return np.zeros((1, 10))  # Return empty feature vector

    def classify_repository(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Perform full classification of repository methodology.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Dictionary with classification results
        """
        repo_name = repo_data.get('info', {}).get('full_name', 'unknown')
        self.logger.info(f"Classifying repository: {repo_name}")
        
        # Validate input data
        if not repo_data:
            self.logger.error("Repository data is empty")
            return {
                'primary_methodology': 'unknown',
                'secondary_methodology': None,
                'confidence_scores': {},
                'raw_scores': {},
                'feature_importance': {}
            }
        
        try:
            # Extract all features
            combined_features = self._extract_all_features(repo_data)
            
            # Calculate methodology scores
            methodology_scores = self._calculate_methodology_scores(combined_features, repo_data)
            
            # Determine primary and secondary methodologies
            sorted_scores = sorted(
                methodology_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Calculate confidence scores
            total_score = sum(methodology_scores.values())
            confidence_scores = {
                method: score / total_score if total_score > 0 else 0
                for method, score in methodology_scores.items()
            }
            
            # Return classification results
            return {
                'primary_methodology': sorted_scores[0][0] if sorted_scores else 'unknown',
                'secondary_methodology': sorted_scores[1][0] if len(sorted_scores) > 1 else None,
                'confidence_scores': confidence_scores,
                'raw_scores': methodology_scores,
                'feature_importance': self._analyze_feature_importance(combined_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying repository {repo_name}: {str(e)}", exc_info=True)
            return {
                'primary_methodology': 'unknown',
                'secondary_methodology': None,
                'confidence_scores': {},
                'raw_scores': {},
                'feature_importance': {}
            }
    def _calculate_methodology_scores(self, features: np.ndarray, repo_data: Dict) -> Dict[str, float]:
        """
        Calculate scores for each methodology based on features.
        
        Args:
            features: Feature array
            repo_data: Repository data dictionary
            
        Returns:
            Dictionary of methodology scores
        """
        scores = defaultdict(float)
        
        # Weight features based on importance for each methodology
        feature_weights = {
            'text_features': 0.3,
            'temporal_features': 0.25,
            'structural_features': 0.25,
            'workflow_features': 0.2
        }
        
        # Special text pattern matching for methodology identification
        text_content = []
        
        # Collect text from issues, PRs, and comments - with proper null checks
        for issue in repo_data.get('issues', []):
            if issue.get('title'):
                text_content.append(issue.get('title', '').lower())
            if issue.get('body'):
                text_content.append(issue.get('body', '').lower())
            
        for pr in repo_data.get('pull_requests', []):
            if pr.get('title'):
                text_content.append(pr.get('title', '').lower())
            if pr.get('body'):
                text_content.append(pr.get('body', '').lower())
            
        for comment in repo_data.get('issue_comments', []) + repo_data.get('pull_request_comments', []):
            if comment.get('body'):
                text_content.append(comment.get('body', '').lower())
        
        # Filter out any None values that might have slipped through
        text_content = [text for text in text_content if text]
        combined_text = ' '.join(text_content)
        
        # Direct pattern matching for methodologies
        for methodology, patterns in self.methodology_patterns.items():
            # Check for terminology matches
            term_matches = sum(combined_text.count(term) for term in patterns.get('terms', []))
            scores[methodology] += term_matches * 0.05
            
            # Check for ceremony mentions
            if 'ceremonies' in patterns:
                ceremony_matches = sum(combined_text.count(ceremony) for ceremony in patterns['ceremonies'])
                scores[methodology] += ceremony_matches * 0.1
                
            # Check for principle mentions
            if 'principles' in patterns:
                principle_matches = sum(combined_text.count(principle) for principle in patterns['principles'])
                scores[methodology] += principle_matches * 0.1
        
        # Calculate time-based patterns
        issues = repo_data.get('issues', [])
        if issues:
            # Find time patterns between issue creation
            issue_dates = []
            for issue in issues:
                created_at = issue.get('created_at')
                if created_at:
                    issue_dates.append(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
            
            if len(issue_dates) > 5:
                sorted_dates = sorted(issue_dates)
                intervals = [(sorted_dates[i+1] - sorted_dates[i]).days 
                             for i in range(len(sorted_dates)-1)]
                
                # Find most common interval patterns
                interval_counts = defaultdict(int)
                for interval in intervals:
                    if interval > 0:  # Ignore same-day issues
                        interval_counts[interval] += 1
                
                # Find matches with methodology time patterns
                for methodology, patterns in self.methodology_patterns.items():
                    for min_days, max_days in patterns.get('time_patterns', []):
                        matching_intervals = sum(1 for interval, count in interval_counts.items() 
                                                if min_days <= interval <= max_days)
                        if matching_intervals > 0:
                            scores[methodology] += 0.15 * matching_intervals
        
        # Calculate weighted scores from extracted features
        feature_index = 0
        for feature_type, weight in feature_weights.items():
            if feature_type == 'text_features':
                feature_length = features.shape[1] // 4  # Approximate division
                feature_score = np.sum(features[:, feature_index:feature_index+feature_length])
                feature_index += feature_length
            elif feature_type == 'temporal_features':
                feature_length = features.shape[1] // 4
                feature_score = np.sum(features[:, feature_index:feature_index+feature_length])
                feature_index += feature_length
            elif feature_type == 'structural_features':
                feature_length = features.shape[1] // 4
                feature_score = np.sum(features[:, feature_index:feature_index+feature_length])
                feature_index += feature_length
            elif feature_type == 'workflow_features':
                feature_length = features.shape[1] - feature_index
                feature_score = np.sum(features[:, feature_index:])
                feature_index += feature_length
            else:
                feature_score = 0
                
            # Distribute feature score across methodologies
            for methodology in self.methodology_patterns.keys():
                scores[methodology] += feature_score * weight / len(self.methodology_patterns)
        
        # If no clear pattern is detected, increase ad_hoc score
        total_pattern_score = sum(scores.values())
        if total_pattern_score < 1.0:
            scores['ad_hoc'] += (1.0 - total_pattern_score) * 2
        
        return dict(scores)
    
    def _safe_text_search(self, text, pattern):
        """
        Safely search for pattern in text handling None values.
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            True if pattern is found, False otherwise
        """
        if text is None or pattern is None:
            return False
        return pattern.lower() in text.lower()
    
    def _analyze_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """
        Analyze which features contributed most to the classification.
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            # Use random forest to calculate feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Create synthetic labels for training
            synthetic_labels = np.zeros(features.shape[0])
            
            # Ensure there are enough samples for training
            if features.shape[0] > 1 and features.shape[1] > 0:
                rf.fit(features, synthetic_labels)
                
                # Get feature importance scores
                importance_scores = rf.feature_importances_
                
                return {
                    f"feature_{i}": float(score)
                    for i, score in enumerate(importance_scores)
                }
            else:
                return {"info": "Not enough samples for feature importance analysis"}
                
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
            return {"error": str(e)}