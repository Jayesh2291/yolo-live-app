"""
Configuration management for YOLO Live App
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model Configuration
    MODEL_PATH: str = "yolov8m.pt"
    CUSTOM_MODEL_PATH: Optional[str] = None
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS: int = 100
    
    # API Configuration
    API_TITLE: str = "YOLO Live Detection API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Real-time object detection API using YOLOv8"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # CORS Configuration
    CORS_ORIGINS: list = ["*"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Configuration
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_model_path() -> Path:
    """Get the active model path"""
    if settings.CUSTOM_MODEL_PATH:
        custom_path = Path(settings.CUSTOM_MODEL_PATH)
        if custom_path.exists():
            return custom_path
    
    # Check in weights directory
    weights_dir = Path(__file__).parent.parent / "weights"
    if weights_dir.exists():
        best_model = weights_dir / "best.pt"
        if best_model.exists():
            return best_model
    
    # Default to yolov8n.pt in backend directory
    return Path(__file__).parent / settings.MODEL_PATH
