import requests
import time
from bs4 import BeautifulSoup

from src.mcp_server.tools.offers_db.data_scraper.utils.pwr_data_processing import parse_polish_date, LocationEnum, ContractTypeEnum
from src.mcp_server.tools.offers_db.data_scraper.scrapers.base_scraper import BaseScraper


class PWRScraper(BaseScraper):
    BASE_URL = "https://biurokarier.pwr.edu.pl/oferty-pracy"
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html",
        "Connection": "keep-alive"
    }

    @staticmethod
    def scrape_offers() -> list[str]:
        """Scrape list of links and ids from job listing pages."""
        results = []
        for page in range(1, 15):
            url = f"{PWRScraper.BASE_URL}/page/{page}/"
            try:
                resp = requests.get(url, headers=PWRScraper.HEADERS)

                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")
                articles = soup.find_all("article", class_="noo_job")

                for art in articles:
                    link_tag = art.select_one("h3.loop-item-title a")
                    if link_tag and "href" in link_tag.attrs:
                        link = link_tag["href"]
                        results.append(link)

            except Exception as e:
                raise Exception(f"Failed to fetch PWR links from page {page}: {e}")
            time.sleep(0.5)

        return results

    @staticmethod
    def scrape_offer_details(offer: dict[str, str]) -> list[dict[str,str]]:
        """Scrape detailed info for each offer given link list."""
        resp = requests.get(offer, headers=PWRScraper.HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title_el = soup.select_one("h1.entry-title")
        company_el = soup.select_one("span.job-company")
        location_el = soup.select_one("span.job-location em")

        title = title_el.get_text(strip=True) if title_el else ""
        company = company_el.get_text(strip=True) if company_el else ""
        location_raw = location_el.get_text(strip=True) if location_el else ""
        loc_enum = LocationEnum.from_raw(location_raw)
        location = loc_enum.value if isinstance(loc_enum, LocationEnum) else loc_enum

        desc_block = soup.find("div", class_="job-desc")
        description = desc_block.get_text(separator="\n", strip=True) if desc_block else ""

        for li in soup.select("div.job-custom-fields li.job-cf"):
            label = li.find("strong")
            value = li.find("span")
            if label and value:
                description += f"\n{label.get_text(strip=True).replace(':', '')}: {value.get_text(strip=True)}"

        tags = [a.get_text(strip=True) for a in soup.select("div.entry-tags a")]
        if tags:
            description += "\n\nTags: " + ", ".join(tags)

        contract_el = soup.select_one("span.job-type span")
        contract_raw = contract_el.get_text(strip=True) if contract_el else ""

        contract_enum = ContractTypeEnum.from_raw(contract_raw)
        contract_type = contract_enum.value if isinstance(contract_enum, ContractTypeEnum) else contract_raw

        posted_el = soup.select_one("span.job-date__posted")
        closing_el = soup.select_one("span.job-date__closing")

        date_posted = parse_polish_date(posted_el.get_text(strip=True)) if posted_el else None
        date_closing = parse_polish_date(closing_el.get_text(strip=True).lstrip("- ")) if closing_el else None

        return {
            "link": offer,
            "title": title,
            "company": company,
            "location": location,
            "contract_type": contract_type,
            "date_posted": date_posted,
            "date_closing": date_closing,
            "description": description,
            "source": "PWR"
        }
