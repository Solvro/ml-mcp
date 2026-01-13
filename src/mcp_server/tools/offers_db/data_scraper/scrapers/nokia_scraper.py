import requests
from datetime import datetime
import re
from bs4 import BeautifulSoup

from src.mcp_server.tools.offers_db.data_scraper.scrapers.base_scraper import BaseScraper


class NokiaScraper(BaseScraper):
    BASE_URL = "https://fa-evmr-saasfaprod1.fa.ocs.oraclecloud.com/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
    JOB_DETAIL_BASE_URL = "https://fa-evmr-saasfaprod1.fa.ocs.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1/job/{id}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    DETAILS_API = (
        "https://fa-evmr-saasfaprod1.fa.ocs.oraclecloud.com/hcmRestApi/resources/latest/"
        "recruitingCEJobRequisitionDetails?expand=all&onlyData=true&finder=ById;Id=\"{id}\",siteNumber=CX_1"
    )
    
    @staticmethod
    def scrape_offers() -> list[str]:
        """Fetch basic info: job ID and link."""
        limit = 200
        offset = 0
        results = []

        finder_value = (
            f"findReqs;siteNumber=CX_1,"
            f"facetsList=LOCATIONS;WORK_LOCATIONS;WORKPLACE_TYPES;TITLES;CATEGORIES;ORGANIZATIONS;POSTING_DATES;FLEX_FIELDS,"
            f"limit={limit},offset={offset},lastSelectedFacet=TITLES,"
            f"locationId=300000000471967,selectedTitlesFacet=TRA,sortBy=POSTING_DATES_DESC"
        )

        params = {
            "onlyData": "true",
            "expand": "requisitionList.workLocation,requisitionList.otherWorkLocations,requisitionList.secondaryLocations,flexFieldsFacet.values,requisitionList.requisitionFlexFields",
            "finder": finder_value
        }

        resp = requests.get(NokiaScraper.BASE_URL, headers=NokiaScraper.HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items or not items[0].get("requisitionList"):
            return results


        for job in items[0]["requisitionList"]:
            title = job['Title'].lower()
            if (('working student' in title or 'summer trainee' in title) and job['PrimaryLocation'] == 'Poland'):
                job_id = str(job["Id"])
                results.append(NokiaScraper.JOB_DETAIL_BASE_URL.format(id=job_id))

            offset += limit

        return results

    @staticmethod
    def scrape_offer_details(offer: str) -> list[dict[str, str]]:
        job_id = NokiaScraper._extract_job_id(offer)
        url = NokiaScraper.DETAILS_API.format(id=job_id)

        resp = requests.get(url, headers=NokiaScraper.HEADERS)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            raise Exception(f"No details found for job {offer}")

        job = items[0]

        title = job.get("Title")
        company = "Nokia"
        location = job.get("PrimaryLocation") or "Poland"
        date_posted = NokiaScraper._parse_date(job.get("ExternalPostedStartDate"))
        date_closing = NokiaScraper._parse_date(job.get("ExternalPostedEndDate"))

        extras = []

        if job.get("JobSchedule"):
            extras.append(f"Schedule: {job['JobSchedule']}")
        if job.get("StudyLevel"):
            extras.append(f"Study level: {job['StudyLevel']}")
        if job.get("ExternalQualificationsStr"):
            extras.append("Qualifications:\n" + NokiaScraper._html_to_text(job["ExternalQualificationsStr"]))
        if job.get("ExternalResponsibilitiesStr"):
            extras.append("Responsibilities:\n" + NokiaScraper._html_to_text(job["ExternalResponsibilitiesStr"]))
        if job.get("OrganizationDescriptionStr"):
            extras.append(NokiaScraper._html_to_text(job["OrganizationDescriptionStr"]))
        if job.get("requisitionFlexFields"):
            for field in job["requisitionFlexFields"]:
                extras.append(f"{field.get('Prompt')}: {field.get('Value')}")

        description = NokiaScraper._html_to_text(job.get("ExternalDescriptionStr", "")) + "\n\n" + "\n\n".join(extras)

        return {
            "title": title,
            "company": company,
            "location": location,
            "link": offer,
            "description": description.strip(),
            "contract_type": "Contract of mandate",
            "date_posted": date_posted,
            "date_closing": date_closing,
            "source": 'Nokia'
        }
    
    @staticmethod
    def _extract_job_id(url: str) -> str:
        match = re.search(r'/job/(\d+)', url)
        if match:
            return match.group(1)
        raise ValueError(f"Job ID not found in {url}")
    
    @staticmethod
    def _html_to_text(html: str) -> str:
        if not html:
            return ""
        return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None