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
    compression_ratio: Optional[float] = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Compression ratio between 0 and 100, where 100 means maximum compression",
    )


class CompressionResponse(BaseModel):
    original_size_kb: float
    compressed_size_kb: float
    compression_ratio: float
    size_reduction_percent: float
    output_image_path: str


def generate_output_path(input_path: str, compression_ratio: float) -> str:
    """Generate output path based on input path and compression ratio."""
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)

    # Split filename into name and extension
    name_parts = filename.split(".")

    if len(name_parts) > 1:
        # Has extension
        name = ".".join(name_parts[:-1])
        extension = "." + name_parts[-1]
    else:
        # No extension
        name = filename
        extension = ""

    # Create new filename with compression ratio
    new_filename = f"{name}_{int(compression_ratio)}{extension}"

    # Combine with directory
    if directory:
        return os.path.join(directory, new_filename)
    else:
        return new_filename


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


async def compress_image(input_path: str, compression_ratio: float) -> dict:
    # Generate output path
    output_path = generate_output_path(input_path, compression_ratio)

    # Compute K from compression_ratio: 0% -> K=256, 100% -> K=1
    K = 256 - int(255 * (compression_ratio / 100))
    # Ensure K is between 1 and 256 (though the formula guarantees this with validation)
    K = max(1, min(256, K))

    logger.info(f"Compressing image {input_path} with K={K} colors")

    loop = asyncio.get_running_loop()

    def load_image():
        with Image.open(input_path) as img:
            return img.convert("RGB")

    image = await loop.run_in_executor(None, load_image)
    image_array = np.array(image, dtype=np.float32)
    pixels = image_array.reshape(-1, 3)

    # Initialize centroids with K-means++
    idx = np.random.choice(len(pixels), min(K, len(pixels)), replace=False)
    centroids = pixels[idx].copy()

    # Run K-means clustering
    for iteration in range(8):
        old_centroids = centroids.copy()
        idx = fast_closest_centroids(pixels, centroids)
        centroids = fast_compute_centroids(pixels, idx, K)

        # Check for convergence
        if np.allclose(old_centroids, centroids, rtol=1e-4):
            logger.info(f"K-means converged after {iteration+1} iterations")
            break

    # Map each pixel to its closest centroid
    idx = fast_closest_centroids(pixels, centroids)
    compressed_pixels = np.array([centroids[i] for i in idx])

    # Reshape back to image dimensions
    compressed_data = compressed_pixels.reshape(image_array.shape)
    compressed_image = Image.fromarray(compressed_data.astype(np.uint8))

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # Save the compressed image
    await loop.run_in_executor(
        None, lambda: compressed_image.save(output_path, quality=85, optimize=True)
    )

    # Calculate compression statistics
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)

    # Calculate size reduction percentage
    size_reduction_percent = (
        ((original_size - compressed_size) / original_size) * 100
        if original_size > 0
        else 0
    )

    logger.info(
        f"Original size: {original_size/1024:.2f} KB, Compressed size: {compressed_size/1024:.2f} KB"
    )
    logger.info(
        f"Size reduction: {size_reduction_percent:.2f}% ({(original_size-compressed_size)/1024:.2f} KB saved)"
    )

    return {
        "original_size_kb": round(original_size / 1024, 2),
        "compressed_size_kb": round(compressed_size / 1024, 2),
        "compression_ratio": (
            round(original_size / compressed_size, 2) if compressed_size > 0 else 1.0
        ),
        "size_reduction_percent": round(size_reduction_percent, 2),
        "output_image_path": output_path,
    }


@app.post("/compress", response_model=CompressionResponse)
async def compress_image_endpoint(request: CompressionRequest):
    if not os.path.isfile(request.input_image_path):
        raise HTTPException(status_code=404, detail="Input image file not found")

    try:
        result = await compress_image(
            request.input_image_path,
            request.compression_ratio,
        )
        # Print the reduction in image size to console as well
        print(f"\nImage Compression Results:")
        print(f"Input image: {request.input_image_path}")
        print(f"Output image: {result['output_image_path']}")
        print(f"Original size: {result['original_size_kb']:.2f} KB")
        print(f"Compressed size: {result['compressed_size_kb']:.2f} KB")
        print(
            f"Size reduction: {result['size_reduction_percent']:.2f}% ({result['original_size_kb'] - result['compressed_size_kb']:.2f} KB saved)"
        )
        print(f"Compression ratio: {result['compression_ratio']:.2f}x\n")

        return CompressionResponse(**result)
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = random.randint(4000, 7000)
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", reload=True)
