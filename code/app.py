import asyncio
import logging
import os
import random
import shutil
import time
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from uvicorn import Config, Server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp_images directory if it doesn't exist
os.makedirs("temp_images", exist_ok=True)

# Initialize FastAPI app and mount static files directory
app = FastAPI()
app.mount("/temp_images", StaticFiles(directory="temp_images"), name="temp_images")

# K-means clustering jokes for logging
KMEANS_JOKES = [
    "Why did K-means go to therapy? It couldn’t find its true center!",
    "K-means walks into a bar and says, 'Cluster me a drink!'",
    "What’s K-means’ favorite dance? The centroid shuffle!",
    "Why was K-means bad at relationships? Too much convergence drama!",
    "K-means tried comedy but got stuck in a local minimum of laughs!",
]

### Helper Functions


def fast_closest_centroids(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute closest centroid indices efficiently using broadcasting."""
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def fast_compute_centroids(pixels: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    """Compute centroids using vectorized operations."""
    return np.array(
        [
            pixels[idx == k].mean(axis=0) if (idx == k).any() else centroids[k]
            for k in range(K)
        ],
        dtype=np.float32,
    )


def generate_unique_path(prefix: str, ext: str) -> str:
    """Generate a unique filepath with prefix and timestamp."""
    timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
    hash_part = random.randint(0, 9999)  # Additional randomness
    filename = f"{prefix}_{timestamp}_{hash_part}{ext}"
    return os.path.join("temp_images", filename)


async def compress_image(input_path: str, compression_ratio: float) -> Dict[str, any]:
    """Compress image using K-means with optimized performance."""
    ext = os.path.splitext(input_path)[1].lower()
    output_path = generate_unique_path("compressed", ext)

    # Calculate number of clusters (K) based on compression ratio
    K = max(1, min(256, 256 - int(255 * (compression_ratio / 100))))
    logger.info(f"Compressing {input_path} with K={K} colors")

    loop = asyncio.get_running_loop()

    # Load image asynchronously in a separate thread
    image = await loop.run_in_executor(
        None, lambda: Image.open(input_path).convert("RGB")
    )
    pixels = np.array(image, dtype=np.float32).reshape(-1, 3)

    # Optimized centroid initialization (sample unique pixels)
    unique_pixels = np.unique(pixels, axis=0)
    idx = np.random.choice(
        len(unique_pixels), min(K, len(unique_pixels)), replace=False
    )
    centroids = unique_pixels[idx].copy()

    # K-means clustering with early stopping
    for i in range(8):
        logger.info(f"Iteration {i+1}: {random.choice(KMEANS_JOKES)}")
        old_centroids = centroids.copy()
        idx = fast_closest_centroids(pixels, centroids)
        centroids = fast_compute_centroids(pixels, idx, K)
        if np.allclose(old_centroids, centroids, rtol=1e-4):
            logger.info(f"K-means converged at iteration {i+1}")
            break

    # Create paletted image efficiently
    palette = centroids.astype(np.uint8).flatten().tolist()
    compressed_image = Image.fromarray(
        idx.astype(np.uint8).reshape(image.size[1], image.size[0]), "P"
    )
    compressed_image.putpalette(palette)

    # Define save parameters based on file extension
    save_params = {
        ".png": {"compress_level": 9},
        ".jpg": {"quality": 50, "optimize": True},
        ".jpeg": {"quality": 50, "optimize": True},
    }
    params = save_params.get(ext, {})
    image_to_save = (
        compressed_image if ext == ".png" else compressed_image.convert("RGB")
    )

    # Save the compressed image asynchronously
    await loop.run_in_executor(None, lambda: image_to_save.save(output_path, **params))

    # Calculate compression statistics
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    size_reduction = (
        ((original_size - compressed_size) / original_size) * 100
        if original_size > 0
        else 0
    )

    return {
        "original_size_kb": round(original_size / 1024, 2),
        "compressed_size_kb": round(compressed_size / 1024, 2),
        "compression_ratio": (
            round(original_size / compressed_size, 2) if compressed_size > 0 else 1.0
        ),
        "size_reduction_percent": round(size_reduction, 2),
        "output_image_path": output_path,
    }


def generate_comparison_html(original_path: str, compressed_path: str) -> str:
    """Generate HTML page to display original and compressed images."""
    original_temp = generate_unique_path("user", os.path.splitext(original_path)[1])
    compressed_temp = compressed_path  # Already in temp_images

    # Copy original image to temp_images directory
    shutil.copy(original_path, original_temp)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Compression Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; }}
            .image-container {{ display: inline-block; margin: 20px; vertical-align: top; }}
            img {{ max-width: 45%; height: auto; display: none; }}
            .loading {{ font-size: 20px; color: #555; }}
            h2 {{ margin: 10px 0; }}
        </style>
        <script>
            window.onload = function() {{
                const images = document.querySelectorAll('img');
                images.forEach(img => {{
                    img.onload = () => {{
                        img.style.display = 'block';
                        img.previousElementSibling.style.display = 'none';
                    }};
                    img.onerror = () => {{
                        img.previousElementSibling.textContent = 'Failed to load image';
                    }};
                }});
            }};
        </script>
    </head>
    <body>
        <h1>Image Compression Comparison</h1>
        <div class="image-container">
            <h2>Original</h2>
            <div class="loading">Loading...</div>
            <img src="/temp_images/{os.path.basename(original_temp)}" alt="Original">
            <p>Size: {os.path.getsize(original_path)/1024:.2f} KB</p>
        </div>
        <div class="image-container">
            <h2>Compressed</h2>
            <div class="loading">Loading...</div>
            <img src="/temp_images/{os.path.basename(compressed_path)}" alt="Compressed">
            <p>Size: {os.path.getsize(compressed_path)/1024:.2f} KB</p>
        </div>
    </body>
    </html>
    """
    return html_content


def write_file(path: str, data: bytes) -> None:
    """Synchronously write bytes to a file."""
    with open(path, "wb") as f:
        f.write(data)


### FastAPI Endpoint


@app.post("/compress/", response_class=HTMLResponse)
async def compress_image_endpoint(
    file: UploadFile = File(...), compression_ratio: float = 50.0
):
    """Handle image upload, compress it, and return comparison HTML."""
    # Validate compression ratio
    if not 0 <= compression_ratio <= 100:
        return HTMLResponse(
            "Compression ratio must be between 0 and 100", status_code=400
        )

    # Generate unique path for uploaded file
    input_path = generate_unique_path("user", os.path.splitext(file.filename)[1])

    # Read the uploaded file content asynchronously
    content = await file.read()  # Returns bytes

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Write the file synchronously in a separate thread to avoid blocking
    await loop.run_in_executor(None, write_file, input_path, content)

    try:
        # Compress the image and get results
        result = await compress_image(input_path, compression_ratio)
        # Generate HTML comparison page
        html_content = generate_comparison_html(input_path, result["output_image_path"])
        return HTMLResponse(content=html_content)
    finally:
        # Optional cleanup (uncomment to remove input file after processing)
        # if os.path.exists(input_path):
        #     os.remove(input_path)
        pass


### Main Server Function


async def main():
    """Start FastAPI server on a random port."""
    port = random.randint(1000, 9000)
    config = Config(app=app, host="0.0.0.0", port=port)
    server = Server(config)
    logger.info(f"Starting server on http://localhost:{port}")
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
