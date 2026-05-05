import os
import sys
import time
import uuid
import shutil
import traceback
from urllib.parse import urlparse, unquote

WORKER_VERSION = "video-generation-r2-v4-quality"
print("BOOT: handler.py starting", flush=True)
print(f"BOOT: worker version: {WORKER_VERSION}", flush=True)

try:
    import runpod
except Exception:
    print("BOOT_FATAL: failed to import runpod", flush=True)
    traceback.print_exc()
    time.sleep(30)
    sys.exit(1)


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
DEFAULT_STEPS = int(os.environ.get("LTX_STEPS", "50"))
HIGH_QUALITY_STEPS = int(os.environ.get("LTX_HIGH_QUALITY_STEPS", "60"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "361"))

PIPE = None


def log(message: str):
    print(message, flush=True)


def require_env(name: str, value: str | None):
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")


def get_s3_client():
    import boto3
    from botocore.config import Config

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
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def public_url_for_key(key: str) -> str:
    return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{key.lstrip('/')}"


def upload_file_to_r2(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    s3 = get_s3_client()

    with open(local_path, "rb") as f:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=f, ContentType=content_type)

    return public_url_for_key(key)


def download_file(url: str, output_path: str) -> str:
    import requests

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return output_path


def extract_user_job_from_image_url(image_url: str, fallback_job_id: str):
    path = unquote(urlparse(image_url).path).lstrip("/")
    parts = path.split("/")

    if len(parts) >= 4 and parts[0] == "uploads":
        return parts[1], parts[2]

    return "unknown-user", fallback_job_id


def output_key_for(user_id: str, job_id: str) -> str:
    return f"outputs/{user_id}/{job_id}/final.mp4"


def dimensions_for(aspect_ratio: str, quality: str = "standard"):
    if quality == "high":
        if aspect_ratio == "9:16":
            return 768, 1344
        if aspect_ratio == "1:1":
            return 1024, 1024
        return 1344, 768

    if aspect_ratio == "9:16":
        return 576, 1024
    if aspect_ratio == "1:1":
        return 768, 768
    return 1024, 576


def frame_count_for(duration: int, fps: int):
    target = int(duration) * fps + 1
    capped = min(target, MAX_VIDEO_FRAMES)
    return capped - ((capped - 1) % 8)


def normalize_image(input_path: str, output_path: str, width: int, height: int):
    from PIL import Image

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

    import torch
    import diffusers
    from diffusers import LTXImageToVideoPipeline

    log(f"BOOT: diffusers={diffusers.__version__}")
    log(f"BOOT: torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the RunPod worker")

    log(f"Loading LTX pipeline from {MODEL_REPO}")

    PIPE = LTXImageToVideoPipeline.from_pretrained(
        MODEL_REPO,
        cache_dir=MODEL_CACHE_DIR,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    PIPE.to("cuda")

    if hasattr(PIPE, "vae") and hasattr(PIPE.vae, "enable_tiling"):
        PIPE.vae.enable_tiling()

    log("LTX pipeline loaded")
    return PIPE


def generate_video(prompt: str, image, output_path: str, duration: int, aspect_ratio: str, quality: str):
    import torch
    from diffusers.utils import export_to_video

    width, height = dimensions_for(aspect_ratio, quality)
    fps = DEFAULT_FPS
    num_frames = frame_count_for(duration, fps)
    steps = HIGH_QUALITY_STEPS if quality == "high" else DEFAULT_STEPS

    negative_prompt = (
        "worst quality, low quality, blurry, jittery, distorted, deformed, "
        "duplicated product, duplicated bottle, extra text, watermark, logo errors, "
        "inconsistent label, bad reflections, flicker, warped geometry, compression artifacts"
    )

    pipe = get_pipeline()

    generator = torch.Generator(device="cuda").manual_seed(
        int.from_bytes(os.urandom(4), "big")
    )

    log(
        f"Generating video: {width}x{height}, frames={num_frames}, "
        f"duration_requested={duration}s, fps={fps}, steps={steps}, quality={quality}"
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
            guidance_scale=5.0 if quality == "high" else 3.5,
            image_cond_noise_scale=0.025,
            generator=generator,
            decode_timestep=0.05 if quality == "high" else 0.03,
            decode_noise_scale=0.025,
        )

    export_to_video(result.frames[0], output_path, fps=fps)

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
        log(f"Job started: {job_id}")
        log(f"Downloading image: {image_url}")

        download_file(image_url, raw_image_path)

        width, height = dimensions_for(aspect_ratio, quality)
        image = normalize_image(raw_image_path, image_path, width, height)

        generate_video(prompt, image, output_path, duration, aspect_ratio, quality)

        user_id, resolved_job_id = extract_user_job_from_image_url(image_url, job_id)
        output_key = output_key_for(user_id, resolved_job_id)

        log(f"Uploading video to R2: {output_key}")
        video_url = upload_file_to_r2(output_path, output_key)

        log(f"Job completed: {job_id}")
        log(f"Video URL: {video_url}")

        return {
            "success": True,
            "video_url": video_url,
            "output_key": output_key,
            "model_repo": MODEL_REPO,
            "duration_requested": duration,
            "frames": frame_count_for(duration, DEFAULT_FPS),
            "fps": DEFAULT_FPS,
            "quality": quality,
        }

    except Exception as e:
        log(f"Job failed: {job_id}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


log("BOOT: starting RunPod serverless worker")
runpod.serverless.start({"handler": handler})
