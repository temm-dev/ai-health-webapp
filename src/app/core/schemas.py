from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str
    filename: str
    analyze: dict
    anti_spoofing_test: Optional[dict] = None
