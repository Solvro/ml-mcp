"""Manual smoke script for the ToPWR API."""

import asyncio

import httpx


async def run_api_smoke() -> None:
    """Run a basic end-to-end smoke check against a live ToPWR API instance."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("Testing ToPWR API...\n")

        print("1) Health endpoint")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")

        print("2) Create conversation")
        chat_request = {
            "user_id": "test_user_123",
            "message": "Czym jest nagroda dziekana?",
            "metadata": {"source": "smoke"},
        }
        response = await client.post(f"{base_url}/api/chat", json=chat_request)
        print(f"   Status: {response.status_code}")
        chat_response = response.json()
        print(f"   Session ID: {chat_response['session_id']}")
        print(f"   Response: {chat_response['message'][:100]}...\n")

        session_id = chat_response["session_id"]

        print("3) Continue conversation")
        continue_request = {
            "user_id": "test_user_123",
            "session_id": session_id,
            "message": "A jakie są wymagania?",
        }
        response = await client.post(f"{base_url}/api/chat", json=continue_request)
        print(f"   Status: {response.status_code}")
        chat_response = response.json()
        print(f"   Message count: {chat_response['metadata']['message_count']}\n")

        print("4) Conversation history")
        response = await client.get(f"{base_url}/api/sessions/{session_id}/history")
        print(f"   Status: {response.status_code}")
        history = response.json()
        print(f"   Total messages: {history['total_messages']}")
        for idx, msg in enumerate(history["messages"], start=1):
            print(f"   [{idx}] {msg['role']}: {msg['content'][:50]}...")
        print()

        print("5) User sessions")
        response = await client.get(f"{base_url}/api/users/test_user_123/sessions")
        print(f"   Status: {response.status_code}")
        user_sessions = response.json()
        print(f"   Session count: {user_sessions['session_count']}\n")

        print("6) System stats")
        response = await client.get(f"{base_url}/api/stats")
        print(f"   Status: {response.status_code}")
        stats = response.json()
        print(f"   Stats: {stats}\n")

        print("7) Session info")
        response = await client.get(f"{base_url}/api/sessions/{session_id}")
        print(f"   Status: {response.status_code}")
        session_info = response.json()
        print(f"   Session Info: {session_info}\n")

        print("Smoke check completed.")


def main() -> None:
    """Entry point for the `api-smoke` console script."""
    asyncio.run(run_api_smoke())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ToPWR API Smoke Check")
    print("=" * 60 + "\n")
    print("Make sure the API server is running:")
    print("  just topwr-api")
    print("  OR")
    print("  uv run topwr-api\n")
    print("=" * 60 + "\n")

    main()
