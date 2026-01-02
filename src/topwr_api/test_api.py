"""Simple test script for ToPWR API."""

import asyncio

import httpx


async def test_api() -> None:
    """Test the ToPWR API endpoints."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("🧪 Testing ToPWR API...\n")

        # 1. Test health endpoint
        print("1️⃣ Testing health endpoint...")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")

        # 2. Create new conversation
        print("2️⃣ Creating new conversation...")
        chat_request = {
            "user_id": "test_user_123",
            "message": "Czym jest nagroda dziekana?",
            "metadata": {"source": "test"},
        }
        response = await client.post(f"{base_url}/api/chat", json=chat_request)
        print(f"   Status: {response.status_code}")
        chat_response = response.json()
        print(f"   Session ID: {chat_response['session_id']}")
        print(f"   Response: {chat_response['message'][:100]}...\n")

        session_id = chat_response["session_id"]

        # 3. Continue conversation
        print("3️⃣ Continuing conversation...")
        continue_request = {
            "user_id": "test_user_123",
            "session_id": session_id,
            "message": "A jakie są wymagania?",
        }
        response = await client.post(f"{base_url}/api/chat", json=continue_request)
        print(f"   Status: {response.status_code}")
        chat_response = response.json()
        print(f"   Message count: {chat_response['metadata']['message_count']}\n")

        # 4. Get conversation history
        print("4️⃣ Getting conversation history...")
        response = await client.get(f"{base_url}/api/sessions/{session_id}/history")
        print(f"   Status: {response.status_code}")
        history = response.json()
        print(f"   Total messages: {history['total_messages']}")
        for i, msg in enumerate(history["messages"], 1):
            print(f"   [{i}] {msg['role']}: {msg['content'][:50]}...")
        print()

        # 5. Get user sessions
        print("5️⃣ Getting user sessions...")
        response = await client.get(f"{base_url}/api/users/test_user_123/sessions")
        print(f"   Status: {response.status_code}")
        user_sessions = response.json()
        print(f"   Session count: {user_sessions['session_count']}\n")

        # 6. Get system stats
        print("6️⃣ Getting system statistics...")
        response = await client.get(f"{base_url}/api/stats")
        print(f"   Status: {response.status_code}")
        stats = response.json()
        print(f"   Stats: {stats}\n")

        # 7. Get session info
        print("7️⃣ Getting session info...")
        response = await client.get(f"{base_url}/api/sessions/{session_id}")
        print(f"   Status: {response.status_code}")
        session_info = response.json()
        print(f"   Session Info: {session_info}\n")

        print("✅ All tests passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ToPWR API Test Suite")
    print("=" * 60 + "\n")
    print("Make sure the API server is running:")
    print("  just topwr-api")
    print("  OR")
    print("  uv run topwr-api\n")
    print("=" * 60 + "\n")

    asyncio.run(test_api())
