import os
from dotenv import load_dotenv
from github import Github
import pandas as pd

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize GitHub client
    g = Github(os.getenv('GITHUB_TOKEN'))
    
    try:
        # Test with a single repository
        repo = g.get_repo("octocat/Hello-World")
        print(f"Repository: {repo.full_name}")
        print(f"Stars: {repo.stargazers_count}")
        print(f"Issues: {repo.open_issues_count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Close GitHub client
        g.close()

if __name__ == "__main__":
    main()