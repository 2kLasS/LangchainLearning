from datetime import datetime

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field, model_validator
from typing import Literal, Optional


class User(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    username: str = Field(..., min_length=2, max_length=10)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., min_length=6, max_length=30)

    gender: Literal["male", "female", "other", "unknown"] = "unknown"
    fans_num: int = Field(default=0, ge=0)
    uid: UUID = Field(default_factory=uuid4, frozen=True)

    @model_validator(mode='before')
    @classmethod
    def forbid_uid(cls, data: dict) -> dict:
        if isinstance(data, dict) and 'uid' in data:
            raise ValueError("uid can not be passed")
        return data

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not '@' in v:
            raise ValueError('Email is not valid')
        return v.strip()

    @computed_field
    def is_influencer(self) -> bool:
        return self.fans_num > 100000


class BlogPost(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    blog_title: str = Field(..., min_length=1, max_length=100)
    blog_content: str = Field(..., min_length=1, max_length=1000)
    author: User

    post_time: datetime = Field(default_factory=datetime.now, frozen=True)

    @model_validator(mode='before')
    @classmethod
    def forbid_time(cls, data: dict) -> dict:
        if isinstance(data, dict) and 'post_time' in data:
            raise ValueError("post_time can not be passed")
        return data


# 实例化写法1
wang = User(username="sh1tuo", age=21, email="idonthavethis@abc.com", gender="male", fans_num=1500)
blog1 = BlogPost(blog_title="原神为什么这么牛逼", blog_content="牛逼在哪", author=wang)

# 第二种写法
blog2 = BlogPost(
    **{
        "blog_title": "原神为什么这么牛逼",
        "blog_content": "牛逼在哪",
        "author": {
            "username": "sh1tuo",
            "age": 21,
            "email": "idonthavethis@abc.com",
            "gender": "male",
            "fans_num": 1500
            }
    }
)
# 或者用函数
blog2 = BlogPost.model_validate(
    {
        "blog_title": "原神为什么这么牛逼",
        "blog_content": "牛逼在哪",
        "author": {
            "username": "sh1tuo",
            "age": 21,
            "email": "idonthavethis@abc.com",
            "gender": "male",
            "fans_num": 1500
            }
    }
)

print(blog2.model_dump_json(indent=2))
