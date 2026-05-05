
WORKER_VERSION = "video-generation-r2-v2"
print(f"Starting worker version: {WORKER_VERSION}")
import os
import uuid
import shutil
from urllib.parse import urlparse, unquote

import boto3
import requests
import runpod
import torch
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video


HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = os.environ.get("MODEL_REPO", "Lightricks/LTX-Video")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/huggingface-cache")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL")

DEFAULT_FPS = int(os.environ.get("VIDEO_FPS", "24"))
DEFAULT_STEPS = int(os.environ.get("LTX_STEPS", "30"))
HIGH_QUALITY_STEPS = int(os.environ.get("LTX_HIGH_QUALITY_STEPS", "45"))

PIPE = None


def require_env(name: str, value: str | None):
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")


def get_s3_client():
    require_env("S3_ENDPOINT", S3_ENDPOINT)
    require_env("S3_ACCESS_KEY", S3_ACCESS_KEY)
    require_env("S3_SECRET_KEY", S3_SECRET_KEY)
    require_env("S3_BUCKET", S3_BUCKET)
    require_env("S3_PUBLIC_BASE_URL", S3_PUBLIC_BASE_URL)

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def public_url_for_key(key: str) -> str:
    return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{key.lstrip('/')}"


def upload_file_to_r2(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    s3 = get_s3_client()

    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=f,
            ContentType=content_type,
        )

    return public_url_for_key(key)


def download_file(url: str, output_path: str) -> str:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return output_path


def extract_user_job_from_image_url(image_url: str, fallback_job_id: str):
    """
    Expected R2 key:
    uploads/{userId}/{jobId}/{timestamp}-{filename}

    Public URL:
    https://pub-xxx.r2.dev/uploads/{userId}/{jobId}/{filename}
    """
    path = unquote(urlparse(image_url).path).lstrip("/")
    parts = path.split("/")

    if len(parts) >= 4 and parts[0] == "uploads":
        return parts[1], parts[2]

    return "unknown-user", fallback_job_id


def output_key_for(user_id: str, job_id: str) -> str:
    return f"outputs/{user_id}/{job_id}/final.mp4"


def dimensions_for(aspect_ratio: str):
    # Keep these modest for first production test. They are multiples of 32.
    if aspect_ratio == "9:16":
        return 480, 704
    if aspect_ratio == "1:1":
        return 512, 512
    return 704, 480


def normalize_image(input_path: str, output_path: str, width: int, height: int):
    image = Image.open(input_path).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    canvas.save(output_path, "PNG")

    return canvas


def get_pipeline():
    global PIPE

    if PIPE is not None:
        return PIPE

    print(f"Loading LTX pipeline from {MODEL_REPO}")

    PIPE = LTXImageToVideoPipeline.from_pretrained(
        MODEL_REPO,
        cache_dir=MODEL_CACHE_DIR,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    PIPE.to("cuda")

    if hasattr(PIPE, "vae") and hasattr(PIPE.vae, "enable_tiling"):
        PIPE.vae.enable_tiling()

    print("LTX pipeline loaded")
    return PIPE


def generate_video(
    prompt: str,
    image: Image.Image,
    output_path: str,
    duration: int,
    aspect_ratio: str,
    quality: str,
):
    width, height = dimensions_for(aspect_ratio)
    fps = DEFAULT_FPS

    # LTX examples commonly use 161 frames. For faster first production tests,
    # use duration * fps, capped to avoid huge first runs.
    num_frames = max(49, min(int(duration) * fps, 121))

    steps = HIGH_QUALITY_STEPS if quality == "high" else DEFAULT_STEPS

    negative_prompt = (
        "worst quality, low quality, blurry, jittery, distorted, deformed, "
        "duplicated product, duplicated bottle, extra text, watermark, logo errors, "
        "inconsistent label, bad reflections, flicker, warped geometry"
    )

    pipe = get_pipeline()

    generator = torch.Generator(device="cuda").manual_seed(
        int.from_bytes(os.urandom(4), "big")
    )

    print(
        f"Generating video: {width}x{height}, frames={num_frames}, "
        f"fps={fps}, steps={steps}, quality={quality}"
    )

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=generator,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
        )

    frames = result.frames[0]
    export_to_video(frames, output_path, fps=fps)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt")
    image_url = job_input.get("image_url")
    duration = int(job_input.get("duration", 5))
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    quality = job_input.get("quality", "standard")
    job_id = job_input.get("jobId") or job_input.get("job_id") or str(uuid.uuid4())

    if not prompt:
        return {"success": False, "error": "Missing prompt"}

    if not image_url:
        return {"success": False, "error": "Missing image_url"}

    work_dir = f"/tmp/job-{uuid.uuid4()}"
    os.makedirs(work_dir, exist_ok=True)

    raw_image_path = os.path.join(work_dir, "input-original")
    image_path = os.path.join(work_dir, "input.png")
    output_path = os.path.join(work_dir, "final.mp4")

    try:
        print(f"Job started: {job_id}")
        print(f"Downloading image: {image_url}")

        download_file(image_url, raw_image_path)

        width, height = dimensions_for(aspect_ratio)
        image = normalize_image(raw_image_path, image_path, width, height)

        generate_video(
            prompt=prompt,
            image=image,
            output_path=output_path,
            duration=duration,
            aspect_ratio=aspect_ratio,
            quality=quality,
        )

        user_id, resolved_job_id = extract_user_job_from_image_url(image_url, job_id)
        final_job_id = job_id or resolved_job_id
        output_key = output_key_for(user_id, final_job_id)

        print(f"Uploading video to R2: {output_key}")
        video_url = upload_file_to_r2(output_path, output_key)

        print(f"Job completed: {job_id}")
        print(f"Video URL: {video_url}")

        return {
            "success": True,
            "video_url": video_url,
            "output_key": output_key,
            "model_repo": MODEL_REPO,
            "received": {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "jobId": job_id,
            },
        }

    except Exception as e:
        print(f"Job failed: {job_id}: {e}")
        return {
            "success": False,
            "error": str(e),
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
