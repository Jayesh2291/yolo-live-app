"""
YOLO model management and inference
"""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from ultralytics import YOLO
from backend.config import settings, get_model_path


logger = logging.getLogger(__name__)


class YOLOModelManager:
    """Manages YOLO model loading and inference"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize model manager
        
        Args:
            model_path: Path to model weights. If None, uses config settings
        """
        self.model_path = model_path or get_model_path()
        self.model: Optional[YOLO] = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model from weights file"""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            
            self.model = YOLO(str(self.model_path)).to('cuda')
            logger.info(f"Model loaded successfully: {self.model_path.name}")
            logger.info(f"Model device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None
    ):
        """
        Run inference on image
        
        Args:
            image: Input image as numpy array
            conf: Confidence threshold (default from settings)
            iou: IOU threshold for NMS (default from settings)
            max_det: Maximum detections (default from settings)
            
        Returns:
            YOLO results object
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        conf = conf or settings.CONFIDENCE_THRESHOLD
        iou = iou or settings.IOU_THRESHOLD
        max_det = max_det or settings.MAX_DETECTIONS
        
        try:
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                max_det=max_det,
                verbose=False
            )
            return results
        
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "task": self.model.task,
            "classes": list(self.model.names.values()) if hasattr(self.model, 'names') else [],
            "num_classes": len(self.model.names) if hasattr(self.model, 'names') else 0,
            "status": "ready"
        }
    
    def reload_model(self, new_model_path: Optional[Path] = None):
        """Reload model with new weights"""
        if new_model_path:
            self.model_path = new_model_path
        self._load_model()


# Global model instance
model_manager: Optional[YOLOModelManager] = None


def get_model_manager() -> YOLOModelManager:
    """Get or create global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = YOLOModelManager()
    return model_manager
