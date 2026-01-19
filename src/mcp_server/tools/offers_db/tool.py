from datetime import date, datetime
from typing import Any, Dict, List

from src.mcp_server.tools.offers_db.data_scraper.offers_db import OffersDBAsync
from src.mcp_server.tools.offers_db.settings import Settings


class Tool:
    """Tool interface for querying offers database."""

    settings = Settings()

    def __init__(self):
        """Initialize the offers database tool."""
        pass

    def _serialize_dates(self, obj):
        """Convert datetime.date and datetime.datetime objects to ISO format strings."""
        if isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_dates(item) for item in obj]
        return obj

    def invoke(
        self,
        internship_info: str,
        include_companies: List[str] | None = None,
        exclude_companies: List[str] | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Execute similarity search on offers database.

        Args:
            internship_info: Free-text description for semantic search
            include_companies: Optional list of company names to include
            exclude_companies: Optional list of company names to exclude
            limit: Maximum number of results to return (default: 5)
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            List of offer dictionaries with serialized dates
        """
        if include_companies:
            include_filters = {'company': include_companies}
        else:
            include_filters = None

        if exclude_companies:
            exclude_filters = {'company': exclude_companies}
        else:
            exclude_filters = None

        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            OffersDBAsync.similarity_search_cosine(
                query=internship_info,
                k=limit,
                offset=offset,
                include_filters=include_filters,
                exclude_filters=exclude_filters
            )
        )

        # Serialize dates to ISO format strings
        return self._serialize_dates(results)

    async def ainvoke(
        self,
        internship_info: str,
        include_companies: List[str] | None = None,
        exclude_companies: List[str] | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Async version of invoke for better performance in concurrent scenarios.

        Args:
            internship_info: Free-text description for semantic search
            include_companies: Optional list of company names to include
            exclude_companies: Optional list of company names to exclude
            limit: Maximum number of results to return (default: 5)
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            List of offer dictionaries with serialized dates
        """
        if include_companies:
            include_filters = {'company': include_companies}
        else:
            include_filters = None

        if exclude_companies:
            exclude_filters = {'company': exclude_companies}
        else:
            exclude_filters = None

        results = await OffersDBAsync.similarity_search_cosine(
            query=internship_info,
            k=limit,
            offset=offset,
            include_filters=include_filters,
            exclude_filters=exclude_filters
        )

        # Serialize dates to ISO format strings
        return self._serialize_dates(results)
