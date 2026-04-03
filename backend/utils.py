"""
Utility functions for YOLO Live App
"""
import logging
import time
from functools import wraps
from typing import Callable
import cv2
import numpy as np
from fastapi import HTTPException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.3f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.3f}s")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> np.ndarray:
    """
    Validate and decode image from bytes
    
    Args:
        image_bytes: Raw image bytes
        max_size: Maximum allowed file size in bytes
        
    Returns:
        Decoded image as numpy array
        
    Raises:
        HTTPException: If image is invalid or too large
    """
    # Check file size
    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )
    
    # Decode image
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported: JPG, PNG, BMP, WEBP"
            )
        
        return img
    
    except Exception as e:
        logger.error(f"Image decoding error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}"
        )


def format_detection_results(results) -> dict:
    """
    Format YOLO detection results into structured JSON
    
    Args:
        results: YOLO model results object
        
    Returns:
        Dictionary with detection information
    """
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3])
                }
            }
            detections.append(detection)
    
    return {
        "detections": detections,
        "count": len(detections),
        "image_shape": {
            "height": results[0].orig_shape[0],
            "width": results[0].orig_shape[1]
        }
    }


import asyncio
