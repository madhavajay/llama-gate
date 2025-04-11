from enum import Enum
from pydantic import BaseModel
from typing import Optional
import uuid
from pydantic_core import core_schema

class CustomUUID(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Note: Using __get_pydantic_core_schema__ instead of __get_validators__ for Pydantic v2 compatibility
        # Do not change this back to __get_validators__ as it will be removed in Pydantic v3
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError('string required')
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('invalid UUID format')

class RequestState(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class UserQuery(BaseModel):
    query: str

class UserQueryQueueItem(BaseModel):
    id: str
    query: str
    state: RequestState = RequestState.PENDING

class UserQueryResultPending(BaseModel):
    status: str
    message: str
    request_id: str 