import asyncio
import logging
import os
import random
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from numba import njit, prange
from PIL import Image
from pydantic import BaseModel, Field

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()


# Request & Response Models
class CompressionRequest(BaseModel):
    input_image_path: str
    output_image_path: str
    compression_ratio: Optional[float] = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Compression ratio between 0 and 100",
    )


class CompressionResponse(BaseModel):
    original_size_kb: float
    compressed_size_kb: float
    compression_ratio: float


@njit(parallel=True, fastmath=True)
def fast_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    idx = np.empty(X.shape[0], dtype=np.int32)
    for i in prange(X.shape[0]):
        idx[i] = np.argmin(np.sum((X[i] - centroids) ** 2, axis=1))
    return idx


@njit(fastmath=True)
def fast_compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    counts = np.zeros(K, dtype=np.int32)

    for i in range(X.shape[0]):
        k = idx[i]
        centroids[k] += X[i]
        counts[k] += 1

    for k in range(K):
        if counts[k] > 0:
            centroids[k] /= counts[k]

    return centroids


async def compress_image(
    input_path: str, output_path: str, compression_ratio: float
) -> dict:
    # Compute K from compression_ratio: 0% -> K=256, 100% -> K=1
    K = 256 - int(255 * (compression_ratio / 100))
    # Ensure K is between 1 and 256 (though the formula guarantees this with validation)
    K = max(1, min(256, K))

    loop = asyncio.get_running_loop()

    def load_image():
        with Image.open(input_path) as img:
            return img.convert("RGB")

    image = await loop.run_in_executor(None, load_image)
    image_array = np.array(image, dtype=np.float32)
    pixels = image_array.reshape(-1, 3)

    idx = np.random.choice(len(pixels), K, replace=False)
    centroids = pixels[idx].copy()

    for _ in range(8):
        old_centroids = centroids.copy()
        idx = fast_closest_centroids(pixels, centroids)
        centroids = fast_compute_centroids(pixels, idx, K)
        if np.allclose(old_centroids, centroids, rtol=1e-4):
            break

    compressed_data = centroids[idx].reshape(image_array.shape)
    compressed_image = Image.fromarray(compressed_data.astype(np.uint8))

    await loop.run_in_executor(
        None, lambda: compressed_image.save(output_path, quality=85, optimize=True)
    )

    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)

    return {
        "original_size_kb": round(original_size / 1024, 2),
        "compressed_size_kb": round(compressed_size / 1024, 2),
        "compression_ratio": (
            round(original_size / compressed_size, 2) if compressed_size > 0 else 1.0
        ),
    }


@app.post("/compress", response_model=CompressionResponse)
async def compress_image_endpoint(request: CompressionRequest):
    if not os.path.isfile(request.input_image_path):
        raise HTTPException(status_code=404, detail="Input image file not found")

    os.makedirs(os.path.dirname(request.output_image_path), exist_ok=True)

    try:
        result = await compress_image(
            request.input_image_path,
            request.output_image_path,
            request.compression_ratio,
        )
        return CompressionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app", host="0.0.0.0", port=random.randint(4000, 7000), log_level="info"
    )
