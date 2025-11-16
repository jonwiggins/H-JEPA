"""
FastAPI Model Server for H-JEPA Inference.

Production-ready REST API for serving H-JEPA models with:
- Feature extraction
- Batch inference
- Health checks
- Prometheus metrics
- Request validation
"""

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from torchvision import transforms

from ..models.hjepa import HJEPA, create_hjepa
from ..utils.checkpoint import load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("hjepa_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("hjepa_request_duration_seconds", "Request latency", ["endpoint"])
INFERENCE_LATENCY = Histogram("hjepa_inference_duration_seconds", "Inference latency")

# Create FastAPI app
app = FastAPI(
    title="H-JEPA Model Server",
    description="REST API for H-JEPA feature extraction and inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


class FeatureRequest(BaseModel):
    """Request model for feature extraction."""

    hierarchy_level: int = Field(default=0, ge=0, le=3, description="Hierarchy level (0-3)")
    return_numpy: bool = Field(default=True, description="Return features as numpy array")


class FeatureResponse(BaseModel):
    """Response model for feature extraction."""

    features: Union[List[List[float]], str]
    shape: List[int]
    hierarchy_level: int
    inference_time_ms: float


class BatchFeatureResponse(BaseModel):
    """Response model for batch feature extraction."""

    features: List[Union[List[List[float]], str]]
    shapes: List[List[int]]
    hierarchy_level: int
    batch_size: int
    total_inference_time_ms: float
    average_inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    device: str
    version: str


class ModelServer:
    """H-JEPA Model Server for inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        img_size: int = 224,
    ) -> None:
        """
        Initialize model server.

        Args:
            model_path: Path to model checkpoint
            device: Device to use (cuda/cpu)
            img_size: Input image size
        """
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[HJEPA] = None
        self.transform = self._create_transform()

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

        logger.info(f"ModelServer initialized on device: {self.device}")

    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transform."""
        return transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self, model_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            model_path: Path to model checkpoint
        """
        logger.info(f"Loading model from: {model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get model config from checkpoint
            config = checkpoint.get("config", {})
            model_config = config.get("model", {})

            # Create model
            self.model = create_hjepa(
                encoder_type=model_config.get("encoder_type", "vit_base_patch16_224"),
                img_size=self.img_size,
                embed_dim=model_config.get("embed_dim", 768),
                predictor_depth=model_config.get("predictor", {}).get("depth", 6),
                predictor_num_heads=model_config.get("predictor", {}).get("num_heads", 12),
                num_hierarchies=model_config.get("num_hierarchies", 3),
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transform
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        hierarchy_level: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract features from image.

        Args:
            image: Preprocessed image tensor [1, C, H, W]
            hierarchy_level: Hierarchy level (0-3)

        Returns:
            Features as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Move to device
        image = image.to(self.device)

        # Extract features
        start_time = time.time()
        features = self.model.extract_features(
            image, level=hierarchy_level, use_target_encoder=True
        )
        inference_time = (time.time() - start_time) * 1000

        # Convert to numpy
        features_np = features.cpu().numpy()

        INFERENCE_LATENCY.observe(inference_time / 1000)

        return features_np, inference_time

    @torch.no_grad()
    def extract_features_batch(
        self,
        images: torch.Tensor,
        hierarchy_level: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract features from batch of images.

        Args:
            images: Batch of preprocessed images [B, C, H, W]
            hierarchy_level: Hierarchy level (0-3)

        Returns:
            Features as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Move to device
        images = images.to(self.device)

        # Extract features
        start_time = time.time()
        features = self.model.extract_features(
            images, level=hierarchy_level, use_target_encoder=True
        )
        inference_time = (time.time() - start_time) * 1000

        # Convert to numpy
        features_np = features.cpu().numpy()

        INFERENCE_LATENCY.observe(inference_time / 1000)

        return features_np, inference_time


# Global model server instance
model_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get or create model server instance."""
    global model_server

    if model_server is None:
        model_path = os.getenv("MODEL_PATH")
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        img_size = int(os.getenv("IMAGE_SIZE", "224"))

        model_server = ModelServer(
            model_path=model_path,
            device=device,
            img_size=img_size,
        )

    return model_server


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize model server on startup."""
    logger.info("Starting H-JEPA Model Server...")

    # Initialize model server
    try:
        server = get_model_server()
        logger.info("Model server ready")
    except Exception as e:
        logger.error(f"Failed to initialize model server: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "H-JEPA Model Server",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    start_time = time.time()

    try:
        server = get_model_server()

        response = HealthResponse(
            status="healthy",
            model_loaded=server.model is not None,
            device=server.device,
            version="1.0.0",
        )

        REQUEST_COUNT.labels(endpoint="health", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="health").observe(time.time() - start_time)

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="health", status="error").inc()
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/extract", response_model=FeatureResponse)
async def extract_features(
    file: UploadFile = File(...),
    hierarchy_level: int = Query(default=0, ge=0, le=3),
    return_numpy: bool = Query(default=True),
) -> FeatureResponse:
    """
    Extract features from a single image.

    Args:
        file: Image file
        hierarchy_level: Hierarchy level (0-3)
        return_numpy: Return as numpy array or list

    Returns:
        Extracted features
    """
    start_time = time.time()

    try:
        server = get_model_server()

        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = server.preprocess_image(image)

        # Extract features
        features, inference_time = server.extract_features(
            image_tensor,
            hierarchy_level=hierarchy_level,
        )

        # Format response
        features_list = features[0].tolist() if not return_numpy else "base64_encoded"

        response = FeatureResponse(
            features=features_list,
            shape=list(features.shape),
            hierarchy_level=hierarchy_level,
            inference_time_ms=inference_time,
        )

        REQUEST_COUNT.labels(endpoint="extract", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="extract").observe(time.time() - start_time)

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="extract", status="error").inc()
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_batch", response_model=BatchFeatureResponse)
async def extract_features_batch(
    files: List[UploadFile] = File(...),
    hierarchy_level: int = Query(default=0, ge=0, le=3),
    return_numpy: bool = Query(default=True),
) -> BatchFeatureResponse:
    """
    Extract features from a batch of images.

    Args:
        files: List of image files
        hierarchy_level: Hierarchy level (0-3)
        return_numpy: Return as numpy array or list

    Returns:
        Extracted features for all images
    """
    start_time = time.time()

    try:
        server = get_model_server()

        # Read and preprocess all images
        image_tensors = []
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_tensor = server.preprocess_image(image)
            image_tensors.append(image_tensor)

        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0)

        # Extract features
        features, inference_time = server.extract_features_batch(
            batch_tensor,
            hierarchy_level=hierarchy_level,
        )

        # Format response
        features_list = [f.tolist() if not return_numpy else "base64_encoded" for f in features]

        response = BatchFeatureResponse(
            features=features_list,
            shapes=[list(f.shape) for f in features],
            hierarchy_level=hierarchy_level,
            batch_size=len(files),
            total_inference_time_ms=inference_time,
            average_inference_time_ms=inference_time / len(files),
        )

        REQUEST_COUNT.labels(endpoint="extract_batch", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="extract_batch").observe(time.time() - start_time)

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="extract_batch", status="error").inc()
        logger.error(f"Batch feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/info")
async def model_info() -> Dict[str, Any]:
    """Get model information."""
    try:
        server = get_model_server()

        if server.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "model_type": "H-JEPA",
            "device": server.device,
            "image_size": server.img_size,
            "num_hierarchies": server.model.num_hierarchies,
            "embed_dim": server.model.embed_dim,
            "num_patches": server.model.get_num_patches(),
            "patch_size": server.model.get_patch_size(),
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
