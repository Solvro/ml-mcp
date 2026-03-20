from abc import ABC, abstractmethod

class BaseScraper(ABC):

    @staticmethod
    @abstractmethod
    def scrape_offers() -> list[str]:
        pass

    @staticmethod
    @abstractmethod
    def scrape_offer_details(offer: str) -> list[dict[str,str]]:
        pass