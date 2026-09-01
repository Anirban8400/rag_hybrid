import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is missing.")

# MongoDB client & collection setup
client = MongoClient(MONGO_URI)
db = client["database12"]
messages_collection = db["chat_messages"]
users_collection = db["users"]
sessions_collection = db["chat_sessions"]

# try:
#     db = client.get_default_database()
# except Exception:
#     db = client["database12"]

def init_db():
    """Ensures indexes required for authenticated users and their chats."""
    users_collection.create_index([("email", ASCENDING)], unique=True)
    users_collection.create_index([("user_id", ASCENDING)], unique=True)
    sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
    sessions_collection.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
    messages_collection.create_index(
        [("user_id", ASCENDING), ("session_id", ASCENDING), ("created_at", ASCENDING)]
    )
    print("Connected to MongoDB & verified user, session, and message indexes.")


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    return users_collection.find_one({"email": email})


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    return users_collection.find_one({"user_id": user_id})


def create_user(user_id: str, email: str, password_hash: str) -> Dict[str, Any]:
    user = {
        "user_id": user_id,
        "email": email,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
    }
    users_collection.insert_one(user)
    return user


def create_session(user_id: str, session_id: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    session = {
        "user_id": user_id,
        "session_id": session_id,
        "label": "New chat",
        "created_at": now,
        "updated_at": now,
    }
    sessions_collection.insert_one(session)
    return session


def get_session(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    return sessions_collection.find_one({"user_id": user_id, "session_id": session_id})


def list_sessions(user_id: str) -> List[Dict[str, Any]]:
    cursor = sessions_collection.find(
        {"user_id": user_id},
        {"_id": 0, "session_id": 1, "label": 1, "created_at": 1, "updated_at": 1},
    ).sort("updated_at", DESCENDING)
    return list(cursor)


def update_session(user_id: str, session_id: str, label: Optional[str] = None):
    update: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
    if label is not None:
        update["label"] = label
    sessions_collection.update_one(
        {"user_id": user_id, "session_id": session_id}, {"$set": update}
    )


def save_message(user_id: str, session_id: str, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
    """Inserts a single message document into MongoDB."""
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources if sources else [],
        "created_at": datetime.now(timezone.utc)
    }
    messages_collection.insert_one(doc)


def get_history(user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Retrieves chronological chat history for a given session_id."""
    cursor = messages_collection.find(
        {"user_id": user_id, "session_id": session_id},
        {"_id": 0, "user_id": 0, "session_id": 0}
    ).sort("created_at", ASCENDING)

    history = []
    for doc in cursor:
        item = {
            "role": doc["role"],
            "content": doc["content"]
        }
        if doc.get("sources"):
            item["sources"] = doc["sources"]
        history.append(item)

    return history
