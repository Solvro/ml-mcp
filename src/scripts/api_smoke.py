"""Manual smoke script for the ToPWR API."""

import asyncio
import logging

import httpx

from src.config.logging_config import configure_logging

logger = logging.getLogger(__name__)


async def run_api_smoke() -> None:
    """Run a basic end-to-end smoke check against a live ToPWR API instance."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        logger.info("Testing ToPWR API at %s", base_url)

        logger.info("1) Health endpoint")
        response = await client.get(f"{base_url}/health")
        logger.info("   Status: %s", response.status_code)
        logger.info("   Response: %s", response.json())

        logger.info("2) Create conversation")
        chat_request = {
            "user_id": "test_user_123",
            "message": "Czym jest nagroda dziekana?",
            "metadata": {"source": "smoke"},
        }
        response = await client.post(f"{base_url}/api/chat", json=chat_request)
        logger.info("   Status: %s", response.status_code)
        chat_response = response.json()
        logger.info("   Session ID: %s", chat_response["session_id"])
        logger.info("   Response: %s...", chat_response["message"][:100])

        session_id = chat_response["session_id"]

        logger.info("3) Continue conversation")
        continue_request = {
            "user_id": "test_user_123",
            "session_id": session_id,
            "message": "A jakie są wymagania?",
        }
        response = await client.post(f"{base_url}/api/chat", json=continue_request)
        logger.info("   Status: %s", response.status_code)
        chat_response = response.json()
        logger.info("   Message count: %s", chat_response["metadata"]["message_count"])

        logger.info("4) Conversation history")
        response = await client.get(f"{base_url}/api/sessions/{session_id}/history")
        logger.info("   Status: %s", response.status_code)
        history = response.json()
        logger.info("   Total messages: %s", history["total_messages"])
        for idx, msg in enumerate(history["messages"], start=1):
            logger.info("   [%d] %s: %s...", idx, msg["role"], msg["content"][:50])

        logger.info("5) User sessions")
        response = await client.get(f"{base_url}/api/users/test_user_123/sessions")
        logger.info("   Status: %s", response.status_code)
        user_sessions = response.json()
        logger.info("   Session count: %s", user_sessions["session_count"])

        logger.info("6) System stats")
        response = await client.get(f"{base_url}/api/stats")
        logger.info("   Status: %s", response.status_code)
        logger.info("   Stats: %s", response.json())

        logger.info("7) Session info")
        response = await client.get(f"{base_url}/api/sessions/{session_id}")
        logger.info("   Status: %s", response.status_code)
        logger.info("   Session Info: %s", response.json())

        logger.info("Smoke check completed.")


def main() -> None:
    """Entry point for the `api-smoke` console script."""
    configure_logging()
    logger.info("ToPWR API smoke check - the API must already be running (just api)")
    asyncio.run(run_api_smoke())


if __name__ == "__main__":
    main()
