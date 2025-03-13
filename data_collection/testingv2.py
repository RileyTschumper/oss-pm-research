import os
from datetime import datetime, timedelta
import pytz  # Add this import
from dotenv import load_dotenv
from github import Github
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_established_repos(github_client, num_repos=10):
    """
    Find well-established repositories meeting criteria:
    - At least 100 stars
    - More than 5 contributors
    - Active for at least 2 years
    - Has regular activity
    """
    # Calculate date 2 years ago - use UTC timezone
    two_years_ago = datetime.now(pytz.UTC) - timedelta(days=2*365)
    
    # Search query
    query = f"stars:>100 created:<{two_years_ago.strftime('%Y-%m-%d')} language:python"
    
    candidates = []
    page = 1
    
    try:
        # Get initial results
        repos = github_client.search_repositories(query=query)
        logging.info(f"Found {repos.totalCount} repositories matching initial criteria")
        
        while len(candidates) < num_repos and page <= 10:  # Limit to 10 pages for testing
            for repo in repos.get_page(page-1):
                try:
                    # Check additional criteria
                    if (repo.get_contributors().totalCount > 5 and 
                        has_recent_activity(repo)):
                        
                        repo_data = {
                            'name': repo.full_name,
                            'stars': repo.stargazers_count,
                            'contributors': repo.get_contributors().totalCount,
                            'created_at': repo.created_at,
                            'last_updated': repo.updated_at
                        }
                        candidates.append(repo_data)
                        logging.info(f"Found qualifying repo: {repo.full_name}")
                        
                        if len(candidates) >= num_repos:
                            break
                            
                except Exception as e:
                    logging.warning(f"Error checking repo {repo.full_name}: {str(e)}")
                    continue
            
            page += 1
            
        # Randomly sample if we found more than needed
        if len(candidates) > num_repos:
            candidates = random.sample(candidates, num_repos)
            
        return candidates
        
    except Exception as e:
        logging.error(f"Error in repository search: {str(e)}")
        return []

def has_recent_activity(repo):
    """Check if repository has had activity in last 3 months"""
    three_months_ago = datetime.now(pytz.UTC) - timedelta(days=90)
    return repo.updated_at > three_months_ago

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize GitHub client
    g = Github(os.getenv('GITHUB_TOKEN'))
    
    try:
        # Get sample repositories
        repos = get_established_repos(g)
        
        # Print results
        print("\nQualifying Repositories:")
        print("-----------------------")
        for i, repo in enumerate(repos, 1):
            print(f"\n{i}. {repo['name']}")
            print(f"   Stars: {repo['stars']}")
            print(f"   Contributors: {repo['contributors']}")
            print(f"   Created: {repo['created_at']}")
            print(f"   Last Updated: {repo['last_updated']}")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        
    finally:
        # Close GitHub client
        g.close()

if __name__ == "__main__":
    main()