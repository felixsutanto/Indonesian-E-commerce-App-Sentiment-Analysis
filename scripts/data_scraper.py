# scripts/data_scraper.py
import pandas as pd
from google_play_scraper import app, reviews, Sort
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppReviewScraper:
    def __init__(self):
        self.apps = {
            'tokopedia': 'com.tokopedia.tkpd',
            'shopee': 'com.shopee.id'
        }
    
    def scrape_reviews(self, app_name, app_id, count=1000):
        """Scrape reviews for a specific app"""
        logger.info(f"Scraping {count} reviews for {app_name}...")
        
        try:
            result, continuation_token = reviews(
                app_id,
                lang='id',  # Indonesian language
                country='id',  # Indonesia
                sort=Sort.NEWEST,
                count=count
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(result)
            df['app_name'] = app_name
            
            logger.info(f"Successfully scraped {len(df)} reviews for {app_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping {app_name}: {str(e)}")
            return pd.DataFrame()
    
    def scrape_all_apps(self, reviews_per_app=1000):
        """Scrape reviews from all configured apps"""
        all_reviews = []
        
        for app_name, app_id in self.apps.items():
            df = self.scrape_reviews(app_name, app_id, reviews_per_app)
            if not df.empty:
                all_reviews.append(df)
            
            # Be respectful to the API
            time.sleep(2)
        
        if all_reviews:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            logger.info(f"Total reviews collected: {len(combined_df)}")
            return combined_df
        else:
            logger.error("No reviews were collected")
            return pd.DataFrame()

if __name__ == "__main__":
    scraper = AppReviewScraper()
    reviews_df = scraper.scrape_all_apps(reviews_per_app=1000)
    
    if not reviews_df.empty:
        reviews_df.to_csv('data/raw/raw_reviews.csv', index=False)
        print(f"Saved {len(reviews_df)} reviews to data/raw/raw_reviews.csv")