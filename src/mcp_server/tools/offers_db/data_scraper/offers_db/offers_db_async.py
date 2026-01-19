"""Async versions of OffersDB methods using asyncpg."""

import asyncio
from typing import Any, Dict, List
from datetime import date

import asyncpg
from langchain_openai import OpenAIEmbeddings

from src.mcp_server.tools.offers_db.settings import Settings


class OffersDBAsync:
    """Async version of OffersDB using asyncpg for PostgreSQL operations."""
    
    settings = Settings()
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY.get_secret_value())

    @staticmethod
    async def _get_connection():
        """Get async database connection."""
        return await asyncpg.connect(
            host=OffersDBAsync.settings.DB_HOST,
            port=OffersDBAsync.settings.DB_PORT,
            database=OffersDBAsync.settings.DB_NAME,
            user=OffersDBAsync.settings.DB_USER,
            password=OffersDBAsync.settings.DB_PASSWORD.get_secret_value()
        )

    @staticmethod
    async def create_vector_index():
        """Create an IVF index on the embedding column using cosine similarity."""
        try:
            conn = await OffersDBAsync._get_connection()
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS offers_embedding_ivfflat_idx
                    ON {OffersDBAsync.settings.OFFERS_TABLE_NAME}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                await conn.execute(f"ANALYZE {OffersDBAsync.settings.OFFERS_TABLE_NAME};")
                print("Vector index created successfully.")
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error creating vector index: {e}")

    @staticmethod
    async def get_current_offers_links(source: str | None = None) -> List[str]:
        """Get current offer links from database."""
        try:
            conn = await OffersDBAsync._get_connection()
            try:
                if source is not None:
                    rows = await conn.fetch(
                        f"SELECT link FROM {OffersDBAsync.settings.OFFERS_TABLE_NAME} WHERE source = $1",
                        source
                    )
                else:
                    rows = await conn.fetch(
                        f"SELECT link FROM {OffersDBAsync.settings.OFFERS_TABLE_NAME}"
                    )
                return [row['link'] for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error fetching current offers: {e}")
            return []

    @staticmethod
    async def add_offer(offer: Dict[str, str]):
        """Add a single offer to the database."""
        try:
            description = offer.get("description", "")
            # Run embedding in thread pool since it's sync
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, OffersDBAsync.embeddings.embed_query, description
            )

            conn = await OffersDBAsync._get_connection()
            try:
                # Use ON CONFLICT to handle duplicates
                await conn.execute(f"""
                    INSERT INTO {OffersDBAsync.settings.OFFERS_TABLE_NAME} (
                        link, title, company, location,
                        contract_type, date_posted, date_closing,
                        source, description, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
                    ON CONFLICT (link) DO UPDATE SET
                        title = EXCLUDED.title,
                        company = EXCLUDED.company,
                        location = EXCLUDED.location,
                        contract_type = EXCLUDED.contract_type,
                        date_posted = EXCLUDED.date_posted,
                        date_closing = EXCLUDED.date_closing,
                        source = EXCLUDED.source,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding
                """,
                    offer.get("link"),
                    offer.get("title"),
                    offer.get("company"),
                    offer.get("location"),
                    offer.get("contract_type"),
                    offer.get("date_posted"),
                    offer.get("date_closing"),
                    offer.get("source"),
                    description,
                    embedding
                )
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error adding offer {offer.get('link', 'unknown')}: {e}")

    @staticmethod
    async def add_offers(offers: List[Dict[str, str]]):
        """Add multiple offers to the database."""
        for offer in offers:
            await OffersDBAsync.add_offer(offer)

    @staticmethod
    async def remove_offer(offer_link: str):
        """Remove a single offer from the database."""
        try:
            conn = await OffersDBAsync._get_connection()
            try:
                await conn.execute(
                    f"DELETE FROM {OffersDBAsync.settings.OFFERS_TABLE_NAME} WHERE link = $1",
                    offer_link
                )
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error removing offer {offer_link}: {e}")

    @staticmethod
    async def remove_offers(offers_links: List[str]):
        """Remove multiple offers from the database."""
        for link in offers_links:
            await OffersDBAsync.remove_offer(link)

    @staticmethod
    async def get_outdated_offers() -> List[str]:
        """Get offers where date_closing has passed."""
        try:
            conn = await OffersDBAsync._get_connection()
            try:
                rows = await conn.fetch(f"""
                    SELECT link
                    FROM {OffersDBAsync.settings.OFFERS_TABLE_NAME}
                    WHERE date_closing IS NOT NULL AND date_closing < $1
                """, date.today())
                return [row['link'] for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error fetching outdated offers: {e}")
            return []

    @staticmethod
    async def similarity_search_cosine(
        query: str,
        k: int = 5,
        offset: int = 0,
        include_filters: Dict[str, List] | None = None,
        exclude_filters: Dict[str, List] | None = None
    ) -> List[Dict]:
        """
        Perform async similarity search using cosine similarity on the embedding column.

        Args:
            query: Text query to embed and search
            k: Number of results to return
            offset: Number of results to skip for pagination
            include_filters: Dict of filters to include
            exclude_filters: Dict of filters to exclude

        Returns:
            List of result dictionaries
        """
        try:
            # Run embedding in thread pool since it's sync
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, OffersDBAsync.embeddings.embed_query, query
            )
            
            where_clauses = []
            params = []
            param_index = 1

            if include_filters:
                for key in ["company", "location", "contract_type", "source"]:
                    if key in include_filters and include_filters[key] and len(include_filters[key]) > 0:
                        values = include_filters[key]
                        placeholders = ",".join([f"${i}" for i in range(param_index, param_index + len(values))])
                        where_clauses.append(f"{key} IN ({placeholders})")
                        params.extend(values)
                        param_index += len(values)

            if exclude_filters:
                for key in ["company", "location", "contract_type", "source"]:
                    if key in exclude_filters and exclude_filters[key] and len(exclude_filters[key]) > 0:
                        values = exclude_filters[key]
                        placeholders = ",".join([f"${i}" for i in range(param_index, param_index + len(values))])
                        where_clauses.append(f"{key} NOT IN ({placeholders})")
                        params.extend(values)
                        param_index += len(values)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            # Add embedding, k, and offset to params
            embedding_param = param_index
            limit_param = param_index + 1
            offset_param = param_index + 2

            sql = f"""
                SELECT id, link, title, company, location, contract_type, date_posted, date_closing, source, description,
                       embedding <=> ${embedding_param}::vector AS distance
                FROM {OffersDBAsync.settings.OFFERS_TABLE_NAME}
                {where_sql}
                ORDER BY distance
                LIMIT ${limit_param} OFFSET ${offset_param}
            """
            params.extend([query_embedding, k, offset])

            conn = await OffersDBAsync._get_connection()
            try:
                rows = await conn.fetch(sql, *params)
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

