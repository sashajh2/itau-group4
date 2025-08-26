import os
import argparse
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import instaloader
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_instagram_session(username: Optional[str] = None, password: Optional[str] = None) -> instaloader.Instaloader:
    """
    Create and optionally authenticate an Instagram session.
    
    Args:
        username: Instagram username for authentication
        password: Instagram password for authentication
        
    Returns:
        Instaloader instance
    """
    loader = instaloader.Instaloader(
        download_videos=True,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False
    )
    
    if username and password:
        try:
            loader.login(username, password)
            logging.info(f"Successfully logged in as {username}")
        except Exception as e:
            logging.warning(f"Failed to login: {e}. Continuing as anonymous user.")
    else:
        logging.info("No credentials provided. Continuing as anonymous user.")
    
    return loader


def scrape_public_account(
    account_name: str,
    output_dir: str,
    max_posts: Optional[int] = None,
    post_filter: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scrape videos from a public Instagram account.
    
    Args:
        account_name: Instagram account username to scrape
        output_dir: Directory to save downloaded content
        max_posts: Maximum number of posts to download (None for all)
        post_filter: Filter posts by type ('VIDEO', 'CAROUSEL_VIDEO', etc.)
        username: Instagram username for authentication
        password: Instagram password for authentication
        
    Returns:
        Dictionary with scraping results and metadata
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup Instagram session
    loader = get_instagram_session(username, password)
    
    try:
        # Get profile
        profile = instaloader.Profile.from_username(loader.context, account_name)
        logging.info(f"Found profile: {profile.full_name} (@{profile.username})")
        logging.info(f"Total posts: {profile.mediacount}")
        
        # Create account-specific directory
        account_dir = output_path / account_name
        account_dir.mkdir(exist_ok=True)
        
        # Save profile metadata
        profile_metadata = {
            "username": profile.username,
            "full_name": profile.full_name,
            "biography": profile.biography,
            "followers": profile.followers,
            "following": profile.followees,
            "total_posts": profile.mediacount,
            "is_private": profile.is_private,
            "is_verified": profile.is_verified,
            "scraped_at": str(profile.created_at) if profile.created_at else None
        }
        
        with open(account_dir / "profile_metadata.json", "w", encoding="utf-8") as f:
            json.dump(profile_metadata, f, indent=2, default=str)
        
        # Download posts
        posts_downloaded = 0
        videos_downloaded = 0
        failed_downloads = []
        
        posts = profile.get_posts()
        if max_posts:
            posts = list(posts)[:max_posts]
        
        for post in tqdm(posts, desc=f"Downloading posts from @{account_name}"):
            try:
                # Apply post filter if specified
                if post_filter and post_filter.upper() not in str(post.typename).upper():
                    continue
                
                # Download the post
                loader.download_post(post, target=account_dir)
                posts_downloaded += 1
                
                # Count videos
                if post.is_video:
                    videos_downloaded += 1
                    
            except Exception as e:
                logging.error(f"Failed to download post {post.shortcode}: {e}")
                failed_downloads.append({
                    "shortcode": post.shortcode,
                    "error": str(e),
                    "timestamp": str(post.date_local) if post.date_local else None
                })
        
        # Save scraping results
        results = {
            "account_name": account_name,
            "scraping_summary": {
                "total_posts_downloaded": posts_downloaded,
                "videos_downloaded": videos_downloaded,
                "failed_downloads": len(failed_downloads),
                "output_directory": str(account_dir.absolute())
            },
            "failed_downloads": failed_downloads,
            "profile_metadata": profile_metadata
        }
        
        with open(account_dir / "scraping_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Scraping completed for @{account_name}")
        logging.info(f"Posts downloaded: {posts_downloaded}")
        logging.info(f"Videos downloaded: {videos_downloaded}")
        logging.info(f"Failed downloads: {len(failed_downloads)}")
        logging.info(f"Output directory: {account_dir}")
        
        return results
        
    except instaloader.exceptions.ProfileNotExistsException:
        logging.error(f"Profile @{account_name} does not exist")
        raise
    except instaloader.exceptions.PrivateProfileNotFollowedException:
        logging.error(f"Profile @{account_name} is private and not followed")
        raise
    except Exception as e:
        logging.error(f"Error scraping @{account_name}: {e}")
        raise


def scrape_multiple_accounts(
    account_names: List[str],
    output_dir: str,
    max_posts_per_account: Optional[int] = None,
    post_filter: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scrape multiple Instagram accounts.
    
    Args:
        account_names: List of Instagram account usernames
        output_dir: Base directory to save downloaded content
        max_posts_per_account: Maximum posts per account
        post_filter: Filter posts by type
        username: Instagram username for authentication
        password: Instagram password for authentication
        
    Returns:
        Dictionary with overall scraping results
    """
    overall_results = {
        "total_accounts": len(account_names),
        "successful_accounts": 0,
        "failed_accounts": 0,
        "account_results": {},
        "overall_summary": {
            "total_posts_downloaded": 0,
            "total_videos_downloaded": 0,
            "total_failed_downloads": 0
        }
    }
    
    for account_name in account_names:
        try:
            logging.info(f"Starting to scrape @{account_name}")
            result = scrape_public_account(
                account_name=account_name,
                output_dir=output_dir,
                max_posts=max_posts_per_account,
                post_filter=post_filter,
                username=username,
                password=password
            )
            
            overall_results["account_results"][account_name] = result
            overall_results["successful_accounts"] += 1
            overall_results["overall_summary"]["total_posts_downloaded"] += result["scraping_summary"]["total_posts_downloaded"]
            overall_results["overall_summary"]["total_videos_downloaded"] += result["scraping_summary"]["videos_downloaded"]
            overall_results["overall_summary"]["total_failed_downloads"] += result["scraping_summary"]["failed_downloads"]
            
        except Exception as e:
            logging.error(f"Failed to scrape @{account_name}: {e}")
            overall_results["failed_accounts"] += 1
            overall_results["account_results"][account_name] = {"error": str(e)}
    
    # Save overall results
    output_path = Path(output_dir)
    with open(output_path / "overall_scraping_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, default=str)
    
    logging.info("Multi-account scraping completed")
    logging.info(f"Successful: {overall_results['successful_accounts']}")
    logging.info(f"Failed: {overall_results['failed_accounts']}")
    
    return overall_results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Scrape videos from public Instagram accounts")
    parser.add_argument("--accounts", nargs="+", required=True, help="Instagram account usernames to scrape")
    parser.add_argument("--output-dir", default="./data/instagram_scraped", help="Output directory for downloaded content")
    parser.add_argument("--max-posts", type=int, help="Maximum number of posts to download per account")
    parser.add_argument("--post-filter", help="Filter posts by type (e.g., 'VIDEO', 'CAROUSEL_VIDEO')")
    parser.add_argument("--username", help="Instagram username for authentication")
    parser.add_argument("--password", help="Instagram password for authentication")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check if instaloader is available
    try:
        import instaloader
    except ImportError:
        logging.error("instaloader is not installed. Please install it with: pip install instaloader")
        return
    
    try:
        if len(args.accounts) == 1:
            # Single account
            result = scrape_public_account(
                account_name=args.accounts[0],
                output_dir=args.output_dir,
                max_posts=args.max_posts,
                post_filter=args.post_filter,
                username=args.username,
                password=args.password
            )
        else:
            # Multiple accounts
            result = scrape_multiple_accounts(
                account_names=args.accounts,
                output_dir=args.output_dir,
                max_posts_per_account=args.max_posts,
                post_filter=args.post_filter,
                username=args.username,
                password=args.password
            )
        
        print(f"\n✅ Scraping completed successfully!")
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
