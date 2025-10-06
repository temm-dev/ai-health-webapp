from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    return "Сервер работает корректно! Health endpoint is OK."