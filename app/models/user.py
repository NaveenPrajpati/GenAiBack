from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, Literal, Any
from bson import ObjectId
import re


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str) and ObjectId.is_valid(v):
            return v
        raise ValueError(f"Invalid ObjectId: {v}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.to_string_ser_schema(),
        )


class UserCreate(BaseModel):
    first_name: str = Field(..., min_length=2, max_length=50)
    last_name: str = Field(..., min_length=2, max_length=50)
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    role: Literal["admin", "user"] = "user"
    password: str = Field(..., min_length=8, max_length=128)
    description: Optional[str] = Field(None, max_length=500)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character")
        return v

    @field_validator("first_name", "last_name")
    @classmethod
    def name_alpha(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z\s\-']+$", v):
            raise ValueError("Name must contain only letters, spaces, hyphens, or apostrophes")
        return v.strip()

    @field_validator("name")
    @classmethod
    def username_clean(cls, v: str) -> str:
        return v.strip()


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: PyObjectId = Field(alias="_id")
    first_name: str
    last_name: str
    name: str
    email: str
    role: str
    description: Optional[str]

    model_config = {"populate_by_name": True}


class LoginResponse(BaseModel):
    message: str
    user: UserResponse
