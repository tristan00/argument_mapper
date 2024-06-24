

from pydantic import BaseModel, Field
from typing import Optional, List


class ContentUnit(BaseModel):
    url: str = Field(..., description="URL")
    source_text: Optional[str] = Field(None, description="source")
    subreddit: Optional[str] = Field(None, description="subreddit")
    author_tag: Optional[str] = Field(None, description="tag")
    tag: Optional[str] = Field(None, description="tag")
    user: Optional[str] = Field(None, description="Username of the author; can be None if deleted or not found")
    text: str = Field(..., description="Text content of the post or comment")
    date: str = Field(..., description="Publication date of the post or comment")
    upvotes: Optional[int] = Field(None, description="Number of upvotes as a string")
    id: str = Field(..., description="Unique identifier of the post or comment")
    parent_id: Optional[str] = Field(None,
                                     description="Parent identifier for comments; for top-level comments, this is the post ID")
    is_post: bool = Field(..., description="Indicator if the instance is a post (True) or a comment (False)")
    title: Optional[str] = Field(None, description="Post title")
    replies: List['ContentUnit'] = Field(list(),
                                         description="List of replies if the instance is a comment")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            'ContentUnit': lambda v: v.dict(),
        }
        from_attributes = True

    def add_reply(self, reply: 'ContentUnit'):
        self.replies.append(reply)

