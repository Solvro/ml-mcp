import requests
import time
import re
import json

from bs4 import BeautifulSoup
from src.mcp_server.tools.offers_db.data_scraper.scrapers.base_scraper import BaseScraper


class SiiScraper(BaseScraper):
    BASE_URL = "https://web-job-api.sii.pl/offers/pl/all/JUNIOR_1,INTERN_3/all/all/all/all/all/all/score/desc/{offset}/{limit}/pl"
    JOB_DETAIL_BASE_URL = "https://sii.pl/oferty-pracy/id/{id}/{title}"

    BASE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }

    SII_EXTRA_HEADERS = {
        "Referer": "https://sii.pl/",
        "Origin": "https://sii.pl"
    }

    @staticmethod
    def scrape_offers() -> list[str]:
        """Scrape job offers from SII API and return minimal info: id and link."""
        limit = 50
        offset = 0
        collected = []

        headers = {**SiiScraper.BASE_HEADERS, **SiiScraper.SII_EXTRA_HEADERS}

        url = SiiScraper.BASE_URL.format(offset=offset, limit=limit)
        response = requests.get(url, headers=headers)

        if response.status_code == 403:
            raise Exception("Access denied (403 Forbidden).")

        response.raise_for_status()
        data = response.json()

        offers = data.get("offers", [])

        if not offers:
            raise Exception("No more offers returned by the SII API.")

        for offer in offers:
            offer_id = str(offer.get("offerId"))
            title = str(offer.get("title")).lower()
            title = re.sub(r'[\s\-–—−]+', '-', title)
            offer_link = SiiScraper.JOB_DETAIL_BASE_URL.format(id=offer_id, title=title)
            collected.append(offer_link)

        offset += limit
        time.sleep(0.3)


        return collected


    @staticmethod
    def scrape_offer_details(offer: str) -> list[dict[str, str]]:
        resp = requests.get(offer, headers=SiiScraper.BASE_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title_input = soup.select_one("input#offer_name")
        title = title_input["value"].strip() if title_input else ""

        company = "Sii"

        loc_div = soup.select_one("div.nsw-m-filter-dropdown")
        location = "Poland"
        if loc_div and loc_div.has_attr("x-data"):
            try:
                xdata_raw = loc_div["x-data"]
                match = re.search(r'locations\s*:\s*(\[.*?\])\s*}', xdata_raw)
                if match:
                    locations_json = json.loads(match.group(1))
                    city = locations_json[0]["locations"][0]["name"]
                    location = SiiScraper._map_location_en(city)
            except Exception:
                location = 'Poland'

        contract_type = None

        description_parts = []

        desc_div = soup.select_one("div.nsw-o-job-add-content__description")
        if desc_div:
            paragraphs = [p.get_text(strip=True) for p in desc_div.find_all("p")]
            description_parts.extend(paragraphs)

        task_header = soup.find("h2", string=lambda s: s and "zadania" in s.lower())
        if task_header:
            task_ul = task_header.find_next("ul")
            if task_ul:
                tasks = [li.get_text(strip=True) for li in task_ul.find_all("li")]
                description_parts.append("Twoje zadania:")
                description_parts.extend([f"- {t}" for t in tasks])

        req_header = soup.find("h2", string=lambda s: s and "wymagania" in s.lower())
        if req_header:
            req_ul = req_header.find_next("ul")
            if req_ul:
                reqs = [li.get_text(strip=True) for li in req_ul.find_all("li")]
                description_parts.append("Wymagania:")
                description_parts.extend([f"- {r}" for r in reqs])

        job_id_el = soup.select_one("p.nsw-o-job-add-content__job-id")
        if job_id_el:
            description_parts.append(job_id_el.get_text(strip=True))

        description = "\n".join(description_parts).strip()

        return {
            "link": offer,
            "title": title,
            "company": company,
            "location": location,
            "contract_type": contract_type,
            "date_posted": None,
            "date_closing": None,
            "description": description,
            "source": "Sii"
        }


    @staticmethod
    def _map_location_en(city: str) -> str:
        """Translate Polish city names to English if needed."""
        pl_to_en = {
            "Warszawa": "Warsaw",
            "Wrocław": "Wroclaw",
            "Kraków": "Krakow",
            "Poznań": "Poznan",
            "Gdańsk": "Gdansk",
            "Łódź": "Lodz",
            "Katowice": "Katowice",
            "Lublin": "Lublin",
            "Rzeszów": "Rzeszow",
            "Remote": "Remote"
        }
        return pl_to_en.get(city.strip(), city)