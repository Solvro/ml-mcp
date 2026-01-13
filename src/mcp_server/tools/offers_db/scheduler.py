import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.mcp_server.tools.offers_db.data_scraper import DataScraper
from src.mcp_server.tools.offers_db.offers_db import OffersDB


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_source(source: str):
    """Process a single source: scrape offers, update database"""
    try:
        current_offers = OffersDB.get_current_offers_links(source)

        new_offers = DataScraper.scrape_offers(source)
        print(f"SCRAPED {source}:", new_offers)

        to_add, to_remove = OffersDB.diff_offers(current_offers, new_offers)
        print(f"TO ADD {source}:", to_add)
        to_add = to_add[:30] if source == 'PWR' else to_add[:10]

        OffersDB.remove_offers(to_remove)

        detailed_offers = DataScraper.scrape_offers_details(source, to_add)
        OffersDB.add_offers(detailed_offers)
        print(f"ADDED {source}:", detailed_offers)

        return {"source": source, "status": "success", "added": len(detailed_offers)}
    except Exception as e:
        print(f"Error processing {source}: {e}")
        return {"source": source, "status": "error", "error": str(e)}

def run_daily_scraping():
    """Run the daily scraping job"""
    try:
        logger.info("Starting daily scraping job...")

        OffersDB.create_vector_index()

        sources = ['Nokia', 'PWR', 'Sii']
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(process_source, source): source 
                for source in sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing {source}: {result}")
                except Exception as e:
                    logger.error(f"Error processing {source}: {e}")
                    results.append({"source": source, "status": "error", "error": str(e)})

        outdated = OffersDB.get_outdated_offers()
        OffersDB.remove_offers(outdated)
        
        logger.info(f"Daily scraping completed. Results: {results}")
    except Exception as e:
        logger.error(f"Error in daily scraping job: {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

def start_scheduler():
    """Start the scheduler with daily scraping job"""
    try:
        scheduler.add_job(
            run_daily_scraping,
            trigger=CronTrigger(hour=2, minute=0),
            id='daily_scraping',
            name='Daily Data Scraping',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Scheduler started successfully. Daily scraping scheduled for 2:00 AM")
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")

def stop_scheduler():
    """Stop the scheduler"""
    try:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")