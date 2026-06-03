from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("Database not connected. Call connect_db() first.")
    return _client


async def connect_db():
    global _client
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI is not set in environment")
    _client = AsyncIOMotorClient(uri)
    # Verify connection
    await _client.admin.command("ping")
    print("Connected to MongoDB Atlas")


async def close_db():
    global _client
    if _client:
        _client.close()
        _client = None
        print("MongoDB connection closed")


def get_db():
    client = get_client()
    db_name = os.getenv("MONGO_URI", "").split("/")[-1].split("?")[0] or "pythonfullstack"
    return client[db_name]
