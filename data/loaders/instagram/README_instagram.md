# Instagram Video Scraper

This module provides functionality to scrape videos from public Instagram accounts using the `instaloader` library.

## Installation

First, install the required dependency:

```bash
pip install instaloader
```

Or install from the project requirements:

```bash
pip install -r requirements.txt
```

## Features

- ✅ Scrape videos from public Instagram accounts
- ✅ Download post metadata and profile information
- ✅ Support for both authenticated and anonymous scraping
- ✅ Filter posts by type (video, carousel, etc.)
- ✅ Limit number of posts per account
- ✅ Batch processing of multiple accounts
- ✅ Comprehensive logging and error handling
- ✅ Progress bars for download tracking

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Scrape a single account (all posts)
python data/loaders/instagram.py --accounts username

# Scrape with post limit
python data/loaders/instagram.py --accounts username --max-posts 50

# Scrape multiple accounts
python data/loaders/instagram.py --accounts account1 account2 account3

# Filter for video posts only
python data/loaders/instagram.py --accounts username --post-filter VIDEO

# Custom output directory
python data/loaders/instagram.py --accounts username --output-dir ./my_instagram_data
```

#### With Authentication

```bash
# Scrape with Instagram login (recommended for better rate limits)
python data/loaders/instagram.py --accounts username --username your_username --password your_password

# Scrape multiple accounts with authentication
python data/loaders/instagram.py --accounts account1 account2 --username your_username --password your_password
```

#### Advanced Options

```bash
# Set log level
python data/loaders/instagram.py --accounts username --log-level DEBUG

# Combine multiple options
python data/loaders/instagram.py \
    --accounts username \
    --max-posts 100 \
    --post-filter VIDEO \
    --output-dir ./videos_only \
    --username your_username \
    --password your_password
```

### Programmatic Usage

#### Single Account Scraping

```python
from data.loaders.instagram import scrape_public_account

# Basic scraping
result = scrape_public_account(
    account_name="username",
    output_dir="./data/instagram_scraped"
)

# With options
result = scrape_public_account(
    account_name="username",
    output_dir="./data/instagram_scraped",
    max_posts=50,
    post_filter="VIDEO",
    username="your_username",
    password="your_password"
)

print(f"Downloaded {result['scraping_summary']['total_posts_downloaded']} posts")
print(f"Videos: {result['scraping_summary']['videos_downloaded']}")
```

#### Multiple Account Scraping

```python
from data.loaders.instagram import scrape_multiple_accounts

result = scrape_multiple_accounts(
    account_names=["account1", "account2", "account3"],
    output_dir="./data/instagram_scraped",
    max_posts_per_account=25,
    post_filter="VIDEO"
)

print(f"Successfully scraped {result['successful_accounts']} accounts")
print(f"Total posts: {result['overall_summary']['total_posts_downloaded']}")
```

## Output Structure

The scraper creates the following directory structure:

```
output_directory/
├── account_name/
│   ├── profile_metadata.json          # Account profile information
│   ├── scraping_results.json          # Scraping summary and results
│   ├── post_shortcode_1.mp4          # Downloaded video files
│   ├── post_shortcode_1.json         # Post metadata
│   ├── post_shortcode_2.mp4
│   ├── post_shortcode_2.json
│   └── ...
└── overall_scraping_results.json      # Summary for multiple accounts
```

### Profile Metadata

```json
{
  "username": "username",
  "full_name": "Full Name",
  "biography": "Bio text",
  "followers": 1000,
  "following": 500,
  "total_posts": 150,
  "is_private": false,
  "is_verified": false,
  "scraped_at": "2024-01-01T00:00:00"
}
```

### Scraping Results

```json
{
  "account_name": "username",
  "scraping_summary": {
    "total_posts_downloaded": 50,
    "videos_downloaded": 30,
    "failed_downloads": 2,
    "output_directory": "/path/to/output"
  },
  "failed_downloads": [
    {
      "shortcode": "ABC123",
      "error": "Network error",
      "timestamp": "2024-01-01T00:00:00"
    }
  ],
  "profile_metadata": { ... }
}
```

## Important Notes

### Rate Limiting

- Instagram has rate limits to prevent abuse
- Using authentication (login) provides higher rate limits
- Anonymous scraping is more limited
- Consider adding delays between requests for large accounts

### Legal and Ethical Considerations

- Only scrape **public** Instagram accounts
- Respect Instagram's Terms of Service
- Don't overload Instagram's servers
- Use scraped data responsibly and ethically
- Consider adding delays between requests

### Technical Limitations

- Cannot scrape private accounts (unless followed)
- Some posts may fail to download due to network issues
- Video quality depends on what Instagram provides
- Large accounts may take significant time to process

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `instaloader` is installed
2. **Profile Not Found**: Check the username spelling
3. **Private Account**: Cannot scrape private accounts
4. **Rate Limiting**: Add delays or use authentication
5. **Network Errors**: Check internet connection

### Debug Mode

Enable debug logging to see detailed information:

```bash
python data/loaders/instagram.py --accounts username --log-level DEBUG
```

### Example Script

Run the example script to see usage patterns:

```bash
python data/loaders/instagram_example.py
```

## Dependencies

- `instaloader`: Instagram scraping library
- `tqdm`: Progress bars
- `pathlib`: Path handling
- Standard library: `os`, `json`, `logging`, `argparse`

## Contributing

When modifying this module:

1. Follow the existing code style
2. Add proper error handling
3. Include logging for debugging
4. Update this README if adding new features
5. Test with both authenticated and anonymous sessions
