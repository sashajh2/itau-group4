#!/usr/bin/env python3
"""
Example script demonstrating how to use the Instagram loader functions.

This script shows how to scrape videos from public Instagram accounts
both programmatically and via command line.
"""

import os
import sys
from pathlib import Path

# Import from the same directory
from .instagram import scrape_public_account, scrape_multiple_accounts


def example_single_account():
    """Example of scraping a single Instagram account."""
    print("🔍 Example: Scraping a single Instagram account")
    
    # Replace with actual Instagram username
    account_name = "example_account"
    output_dir = "./data/instagram_scraped"
    
    try:
        result = scrape_public_account(
            account_name=account_name,
            output_dir=output_dir,
            max_posts=10,  # Limit to 10 posts for testing
            post_filter="VIDEO"  # Only download video posts
        )
        
        print(f"✅ Successfully scraped @{account_name}")
        print(f"Posts downloaded: {result['scraping_summary']['total_posts_downloaded']}")
        print(f"Videos downloaded: {result['scraping_summary']['videos_downloaded']}")
        
    except Exception as e:
        print(f"❌ Failed to scrape @{account_name}: {e}")


def example_multiple_accounts():
    """Example of scraping multiple Instagram accounts."""
    print("\n🔍 Example: Scraping multiple Instagram accounts")
    
    # Replace with actual Instagram usernames
    account_names = ["account1", "account2", "account3"]
    output_dir = "./data/instagram_scraped"
    
    try:
        result = scrape_multiple_accounts(
            account_names=account_names,
            output_dir=output_dir,
            max_posts_per_account=5,  # Limit to 5 posts per account
            post_filter="VIDEO"  # Only download video posts
        )
        
        print(f"✅ Successfully scraped {result['successful_accounts']} accounts")
        print(f"Total posts downloaded: {result['overall_summary']['total_posts_downloaded']}")
        print(f"Total videos downloaded: {result['overall_summary']['total_videos_downloaded']}")
        
    except Exception as e:
        print(f"❌ Failed to scrape accounts: {e}")


def example_with_authentication():
    """Example of scraping with Instagram authentication."""
    print("\n🔍 Example: Scraping with authentication")
    
    # Replace with actual credentials
    username = "your_instagram_username"
    password = "your_instagram_password"
    account_name = "example_account"
    output_dir = "./data/instagram_scraped"
    
    try:
        result = scrape_public_account(
            account_name=account_name,
            output_dir=output_dir,
            max_posts=20,
            username=username,
            password=password
        )
        
        print(f"✅ Successfully scraped @{account_name} with authentication")
        print(f"Posts downloaded: {result['scraping_summary']['total_posts_downloaded']}")
        
    except Exception as e:
        print(f"❌ Failed to scrape @{account_name}: {e}")


def main():
    """Main function to run examples."""
    print("📱 Instagram Scraper Examples")
    print("=" * 40)
    
    # Check if instaloader is available
    try:
        import instaloader
        print("✅ instaloader is available")
    except ImportError:
        print("❌ instaloader is not installed. Please install it first:")
        print("   pip install instaloader")
        return
    
    print("\nNote: These examples use placeholder account names.")
    print("Replace them with actual Instagram usernames to test.")
    
    # Run examples
    example_single_account()
    example_multiple_accounts()
    example_with_authentication()
    
    print("\n" + "=" * 40)
    print("📖 Usage Examples:")
    print("\nCommand line usage:")
    print("  # Scrape a single account")
    print("  python instagram.py --accounts example_account --max-posts 10")
    print("\n  # Scrape multiple accounts")
    print("  python instagram.py --accounts account1 account2 --max-posts 5")
    print("\n  # With authentication")
    print("  python instagram.py --accounts example_account --username your_user --password your_pass")
    print("\n  # Filter for video posts only")
    print("  python instagram.py --accounts example_account --post-filter VIDEO")
    
    print("\nProgrammatic usage:")
    print("  from instagram import scrape_public_account")
    print("  result = scrape_public_account('username', './output')")


if __name__ == "__main__":
    main()
