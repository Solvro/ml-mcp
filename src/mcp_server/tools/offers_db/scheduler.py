import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.mcp_server.tools.offers_db.data_scraper import DataScraper
from src.mcp_server.tools.offers_db.data_scraper.offers_db import OffersDBAsync


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def process_source(source: str):
    """Process a single source: scrape offers, update database (async)"""
    try:
        current_offers = await OffersDBAsync.get_current_offers_links(source)

        new_offers = DataScraper.scrape_offers(source)
        logger.info(f"SCRAPED {source}: {len(new_offers)} offers")

        # Calculate diff
        to_add = set(new_offers) - set(current_offers)
        to_remove = set(current_offers) - set(new_offers)
        to_add = list(to_add)
        to_remove = list(to_remove)
        
        logger.info(f"TO ADD {source}: {len(to_add)} offers")
        to_add = to_add[:30] if source == 'PWR' else to_add[:10]

        if to_remove:
            await OffersDBAsync.remove_offers(to_remove)

        detailed_offers = DataScraper.scrape_offers_details(source, to_add)
        await OffersDBAsync.add_offers(detailed_offers)
        logger.info(f"ADDED {source}: {len(detailed_offers)} offers")

        return {"source": source, "status": "success", "added": len(detailed_offers)}
    except Exception as e:
        logger.error(f"Error processing {source}: {e}")
        return {"source": source, "status": "error", "error": str(e)}

async def run_daily_scraping():
    """Run the daily scraping job (async)"""
    try:
        logger.info("Starting daily scraping job...")

        await OffersDBAsync.create_vector_index()

        sources = ['Nokia', 'PWR', 'Sii']
        
        # Process sources concurrently
        tasks = [process_source(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {sources[i]}: {result}")
                processed_results.append({"source": sources[i], "status": "error", "error": str(result)})
            else:
                processed_results.append(result)
                logger.info(f"Completed processing {result.get('source', 'unknown')}: {result}")

        outdated = await OffersDBAsync.get_outdated_offers()
        if outdated:
            await OffersDBAsync.remove_offers(outdated)

        logger.info(f"Daily scraping completed. Results: {processed_results}")
    except Exception as e:
        logger.error(f"Error in daily scraping job: {e}", exc_info=True)

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
