from enum import Enum
from datetime import datetime
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MonthPL(Enum):
    stycznia = 1
    lutego = 2
    marca = 3
    kwietnia = 4
    maja = 5
    czerwca = 6
    lipca = 7
    sierpnia = 8
    września = 9
    października = 10
    listopada = 11
    grudnia = 12

def parse_polish_date(date_str: str) -> datetime | None:
    """Convert '1 lipca 2025' to datetime(2025, 7, 1) using MonthPL Enum."""
    try:
        parts = date_str.strip().split()
        if len(parts) == 3:
            day = int(parts[0])
            month = MonthPL[parts[1]].value
            year = int(parts[2])
            return datetime(year, month, day)
    except (ValueError, KeyError) as e:
        logging.warning(f"Could not parse date: '{date_str}': {e}")
    return None

from enum import Enum


class LocationEnum(str, Enum):
    POLAND = "Poland"
    WROCLAW = "Wroclaw"
    ABROAD = "Abroad"

    @classmethod
    def from_raw(cls, value: str) -> "LocationEnum | str":
        mapping = {
            "Polska": cls.POLAND,
            "Wrocław": cls.WROCLAW,
            "za granicą": cls.ABROAD,
        }
        return mapping.get(value.strip(), value.strip())


class ContractTypeEnum(str, Enum):
    B2B = "B2B"
    SELF_EMPLOYED = "Self-employed"
    STUDENT_INTERNSHIP = "Student internship"
    EMPLOYMENT_CONTRACT = "Employment contract"
    INTERNSHIP_CONTRACT = "Internship/traineeship contract"
    VOLUNTEER_CONTRACT = "Volunteering contract"
    MANDATE_OR_SPECIFIC = "Contract of mandate or specific-task"

    @classmethod
    def from_raw(cls, value: str) -> "ContractTypeEnum | str":
        value = value.strip().lower()
        mapping = {
            "b2b": cls.B2B,
            "b2b-2": cls.B2B,
            "samozatrudnienie": cls.SELF_EMPLOYED,
            "studencka praktyka zawodowa": cls.STUDENT_INTERNSHIP,
            "umowa o pracę": cls.EMPLOYMENT_CONTRACT,
            "umowa o praktyke staz": cls.INTERNSHIP_CONTRACT,
            "umowa o praktykę lub staż": cls.INTERNSHIP_CONTRACT,
            "umowa o wolontariat": cls.VOLUNTEER_CONTRACT,
            "umowa zlecenie, umowa o dzieło": cls.MANDATE_OR_SPECIFIC,
        }
        return mapping.get(value, value)
