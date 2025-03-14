import asyncio
import logging
import os
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from numba import njit, prange
from PIL import Image
from pydantic import BaseModel

# Configure logging with minimal overhead
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

# Constants
_BUFFER_SIZE = 1024 * 1024  # 1MB buffer
_TEMP_BUFFER = np.zeros((_BUFFER_SIZE, 3), dtype=np.float32)

app = FastAPI(title="Image Compression API", openapi_url=None)


class CompressionRequest(BaseModel):
    input_image_path: str
    output_image_path: str
    compression_level: Optional[int] = 20

    class Config:
        schema_extra = {
            "example": {
                "input_image_path": "input.jpg",
                "output_image_path": "output/compressed.jpg",
                "compression_level": 20,
            }
        }


class CompressionResponse(BaseModel):
    original_path: str
    compressed_path: str
    original_size_kb: float
    compressed_size_kb: float
    compression_ratio: float


@njit(parallel=True, fastmath=True, cache=True)
def fast_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    idx = np.empty(m, dtype=np.int32)
    for i in prange(m):
        min_dist = np.inf
        best_idx = 0
        for j in range(centroids.shape[0]):
            dist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - centroids[j, k]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        idx[i] = best_idx
    return idx


@njit(fastmath=True, cache=True)
def fast_compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    n = X.shape[1]
    centroids = np.zeros((K, n), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int32)

    for i in range(X.shape[0]):
        k = idx[i]
        for j in range(n):
            centroids[k, j] += X[i, j]
        counts[k] += 1

    for k in range(K):
        if counts[k] > 0:
            inv_count = 1.0 / counts[k]
            for j in range(n):
                centroids[k, j] *= inv_count
    return centroids


async def compress_image(input_path: str, output_path: str, k_value: int) -> dict:
    if not (1 <= k_value <= 256):
        raise ValueError("Compression level must be between 1 and 256")

    logger.info(f"Compressing {input_path} with K={k_value}")
    try:
        loop = asyncio.get_running_loop()

        # Async image loading
        def load_image():
            with Image.open(input_path) as img:
                return img.convert("RGB")

        image = await loop.run_in_executor(None, load_image)
        image_array = np.array(image, dtype=np.float32)
        h, w, d = image_array.shape

        pixels = h * w
        X = (
            _TEMP_BUFFER[:pixels]
            if pixels <= _BUFFER_SIZE
            else np.zeros((pixels, d), dtype=np.float32)
        )
        np.reshape(image_array, (pixels, d), out=X)

        # Optimized K-means initialization
        idx = np.random.choice(pixels, k_value, replace=False)
        centroids = X[idx].copy()

        # K-means with early stopping
        max_iter = 8
        for _ in range(max_iter):
            old_centroids = centroids.copy()
            idx = fast_closest_centroids(X, centroids)
            centroids = fast_compute_centroids(X, idx, k_value)
            if np.allclose(old_centroids, centroids, rtol=1e-4):
                break

        # Efficient reconstruction
        compressed_data = centroids[idx].reshape(h, w, d)
        compressed_image = Image.fromarray(compressed_data.astype(np.uint8))

        # Async save with quality optimization
        await loop.run_in_executor(
            None, lambda: compressed_image.save(output_path, quality=85, optimize=True)
        )

        # File stats
        original_size = os.path.getsize(input_path) / 1024
        compressed_size = os.path.getsize(output_path) / 1024

        return {
            "original_size": round(original_size, 2),
            "compressed_size": round(compressed_size, 2),
            "compression_ratio": round(original_size / compressed_size, 2),
        }

    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise


@app.post("/compress", response_model=CompressionResponse)
async def compress_image_endpoint(request: CompressionRequest):
    input_path = os.path.abspath(request.input_image_path)
    output_path = os.path.abspath(request.output_image_path)

    if not os.path.isfile(input_path):
        raise HTTPException(status_code=404, detail="Input image file not found")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        result = await compress_image(
            input_path, output_path, request.compression_level
        )
        return CompressionResponse(
            original_path=input_path, compressed_path=output_path, **result
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


# Warmup Numba functions
@njit(cache=True)
def warmup():
    X = np.random.rand(100, 3).astype(np.float32)
    centroids = X[:5].copy()
    idx = fast_closest_centroids(X, centroids)
    fast_compute_centroids(X, idx, 5)


if __name__ == "__main__":
    warmup()
    logger.info("Starting Image Compression API")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,  # Fixed port for consistency
        log_level="error",
        workers=min(os.cpu_count() or 1, 4),  # Optimal worker count
        timeout_keep_alive=30,
    )
