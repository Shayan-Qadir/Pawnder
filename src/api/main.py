"""
Pawnder (PressureLab) - FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.routes.user import router as user_router
from src.api.routes.scenario import router as scenario_router
from src.api.routes.practice import router as practice_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="MLOps-driven chess practice predicting difficult scenarios from player game history.",
    version="1.0.0",
)

# Enable CORS for local & frontend UI interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(user_router, prefix=settings.API_PREFIX)
app.include_router(scenario_router, prefix=settings.API_PREFIX)
app.include_router(practice_router, prefix=settings.API_PREFIX)


@app.get("/health")
def healthcheck():
    """Healthcheck endpoint for Docker, CloudWatch, and ALB."""
    return {"status": "healthy", "service": "Pawnder API", "version": "1.0.0"}
