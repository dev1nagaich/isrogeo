from motor.motor_asyncio import AsyncIOMotorClient
import os

# CRITICAL: Get MongoDB URL from environment
# Modal injects secrets as environment variables
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "geonli_db")

# Validate MongoDB URL
if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is not set! Check your Modal secrets.")

if MONGODB_URL == "mongodb://localhost:27017":
    raise ValueError("MONGODB_URL is set to localhost! You need to use MongoDB Atlas URL in Modal secrets.")

print(f"📍 MongoDB URL configured: {MONGODB_URL[:20]}...")
print(f"📍 Database name: {DATABASE_NAME}")

# Collection names
USERS_COLLECTION = "users"
SESSIONS_COLLECTION = "sessions"
PROJECTS_COLLECTION = "projects"
MESSAGES_COLLECTION = "messages"
IMAGES_COLLECTION = "images"

_client = None
_database = None

def get_client():
    global _client
    if _client is None:
        print(f"🔗 Connecting to MongoDB...")
        _client = AsyncIOMotorClient(MONGODB_URL)
    return _client

def get_database():
    global _database
    if _database is None:
        client = get_client()
        _database = client[DATABASE_NAME]
    return _database

async def close_database():
    global _client, _database
    if _client:
        _client.close()
        _client = None
        _database = None
        print("🔌 MongoDB connection closed")

async def init_database():
    """Initialize database indexes"""
    db = get_database()
    
    print("📊 Creating database indexes...")
    
    # Users collection
    await db[USERS_COLLECTION].create_index("email", unique=True)
    await db[USERS_COLLECTION].create_index("createdAt")
    print(f"  ✅ {USERS_COLLECTION} indexes created")
    
    # Sessions collection
    await db[SESSIONS_COLLECTION].create_index([("userId", 1), ("updatedAt", -1)])
    await db[SESSIONS_COLLECTION].create_index("projectId")
    await db[SESSIONS_COLLECTION].create_index("archived")
    print(f"  ✅ {SESSIONS_COLLECTION} indexes created")
    
    # Projects collection
    await db[PROJECTS_COLLECTION].create_index([("userId", 1), ("createdAt", -1)])
    print(f"  ✅ {PROJECTS_COLLECTION} indexes created")
    
    # Messages collection
    await db[MESSAGES_COLLECTION].create_index([("sessionId", 1), ("createdAt", 1)])
    await db[MESSAGES_COLLECTION].create_index("userId")
    await db[MESSAGES_COLLECTION].create_index("sender")
    print(f"  ✅ {MESSAGES_COLLECTION} indexes created")
    
    # Images collection
    await db[IMAGES_COLLECTION].create_index([("sessionId", 1), ("uploadedAt", -1)])
    await db[IMAGES_COLLECTION].create_index("userId")
    print(f"  ✅ {IMAGES_COLLECTION} indexes created")
    
    print("✅ All database indexes created successfully")