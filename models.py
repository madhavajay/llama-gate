import uuid
from pydantic import BaseModel

class CustomUUID(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        try:
            return str(uuid.UUID(v))
        except ValueError:
            raise ValueError("Invalid UUID format")

class UserQuery(BaseModel):
    query: str

class UserQueryQueueItem(BaseModel):
    id: CustomUUID
    query: str

class UserQueryResultPending(BaseModel):
    status: str
    message: str
    request_id: CustomUUID 