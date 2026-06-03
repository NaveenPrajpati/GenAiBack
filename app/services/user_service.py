import hashlib
import os
from bson import ObjectId
from app.models.user import UserCreate, UserLogin
from app.database import get_db


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return key.hex() == key_hex


def _collection():
    return get_db()["users"]


async def get_all_users() -> list[dict]:
    users = await _collection().find({}, {"password_hash": 0}).to_list(length=None)
    return users


async def get_user_by_id(user_id: str) -> dict | None:
    if not ObjectId.is_valid(user_id):
        return None
    return await _collection().find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})


async def signup_user(user: UserCreate) -> dict:
    col = _collection()
    if await col.find_one({"email": user.email}):
        raise ValueError("Email already registered")
    record = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "description": user.description,
        "password_hash": _hash_password(user.password),
    }
    result = await col.insert_one(record)
    return await col.find_one({"_id": result.inserted_id}, {"password_hash": 0})


async def login_user(credentials: UserLogin) -> dict:
    col = _collection()
    user = await col.find_one({"email": credentials.email})
    if not user or not _verify_password(credentials.password, user["password_hash"]):
        raise ValueError("Invalid email or password")
    user.pop("password_hash", None)
    return user
