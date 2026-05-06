import os
import sys
import time
import uuid
import shutil
import traceback
from math import sqrt
from urllib.parse import urlparse, unquote

WORKER_VERSION = "wan2.2-i2v-a14b-r2-v1"

MODEL_REPO = os.environ.get("MODEL_REPO", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/huggingface-cache")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(MODEL_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_CACHE_DIR, "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("BOOT: handler.py starting", flush=True)
print(f"BOOT: worker version: {WORKER_VERSION}", flush=True)
print(f"BOOT: model repo: {MODEL_REPO}", flush=True)
print(f"BOOT: model cache: {MODEL_CACHE_DIR}", flush=True)

try:
    import runpod
    import requests
    import boto3
    import numpy as np
    import torch
    from PIL import Image
    from botocore.config import Config
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    print("BOOT: imports successful", flush=True)
    print(
        f"BOOT: torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}",
        flush=True,
    )
except Exception:
    print("BOOT_IMPORT_ERROR:", flush=True)
    traceback.print_exc()
    time.sleep(30)
    sys.exit(1)

HF_TOKEN = os.environ.get("HF_TOKEN")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL")

WAN_FPS = int(os.environ.get("WAN_FPS", "16"))
WAN_STEPS = int(os.environ.get("WAN_STEPS", "40"))
WAN_HIGH_STEPS = int(os.environ.get("WAN_HIGH_STEPS", "50"))
WAN_GUIDANCE_SCALE = float(os.environ.get("WAN_GUIDANCE_SCALE", "3.5"))
WAN_STANDARD_MAX_AREA = int(os.environ.get("WAN_STANDARD_MAX_AREA", str(480 * 832)))
WAN_HIGH_MAX_AREA = int(os.environ.get("WAN_HIGH_MAX_AREA", str(720 * 1280)))
WAN_CPU_OFFLOAD = os.environ.get("WAN_CPU_OFFLOAD", "true").lower() == "true"

PIPE = None


def log(message: str):
    print(message, flush=True)


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
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def public_url_for_key(key: str) -> str:
    return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{key.lstrip('/')}"


def upload_file_to_r2(local_path: str, key: str) -> str:
    s3 = get_s3_client()
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=f, ContentType="video/mp4")
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
    path = unquote(urlparse(image_url).path).lstrip("/")
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "uploads":
        return parts[1], parts[2]
    return "unknown-user", fallback_job_id


def output_key_for(user_id: str, job_id: str) -> str:
    return f"outputs/{user_id}/{job_id}/final.mp4"


def aspect_to_ratio(aspect_ratio: str) -> float:
    if aspect_ratio == "9:16":
        return 9 / 16
    if aspect_ratio == "1:1":
        return 1
    return 16 / 9


def crop_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    current_ratio = image.width / image.height

    if current_ratio > target_ratio:
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        return image.crop((left, 0, left + new_width, image.height))

    new_height = int(image.width / target_ratio)
    top = (image.height - new_height) // 2
    return image.crop((0, top, image.width, top + new_height))


def get_mod_value(pipe) -> int:
    return int(pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1])


def dimensions_for(pipe, aspect_ratio: str, quality: str):
    target_ratio = aspect_to_ratio(aspect_ratio)
    max_area = WAN_HIGH_MAX_AREA if quality == "high" else WAN_STANDARD_MAX_AREA
    h_over_w = 1 / target_ratio
    mod_value = get_mod_value(pipe)

    height = max(mod_value, round(sqrt(max_area * h_over_w)) // mod_value * mod_value)
    width = max(mod_value, round(sqrt(max_area / h_over_w)) // mod_value * mod_value)

    return width, height


def prepare_image(input_path: str, pipe, aspect_ratio: str, quality: str) -> Image.Image:
    image = Image.open(input_path).convert("RGB")
    width, height = dimensions_for(pipe, aspect_ratio, quality)
    image = crop_to_ratio(image, width / height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def frame_count_for(duration: int, fps: int) -> int:
    frames = max(1, int(duration)) * fps + 1
    # Wan examples use 81 frames for ~5 sec at 16fps: (frames - 1) divisible by 4.
    frames = frames - ((frames - 1) % 4)
    return max(17, frames)


def get_pipeline():
    global PIPE

    if PIPE is not None:
        return PIPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the RunPod worker")

    torch.backends.cuda.matmul.allow_tf32 = True

    log(f"Loading Wan pipeline from {MODEL_REPO}")

    PIPE = WanImageToVideoPipeline.from_pretrained(
        MODEL_REPO,
        cache_dir=MODEL_CACHE_DIR,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    if WAN_CPU_OFFLOAD:
        log("Using model CPU offload")
        PIPE.enable_model_cpu_offload()
    else:
        PIPE.to("cuda")

    log("Wan pipeline loaded")
    return PIPE


def generate_video(prompt: str, image: Image.Image, output_path: str, duration: int, quality: str):
    pipe = get_pipeline()

    width = image.width
    height = image.height
    num_frames = frame_count_for(duration, WAN_FPS)
    steps = WAN_HIGH_STEPS if quality == "high" else WAN_STEPS

    negative_prompt = (
        "low quality, worst quality, blurry, overexposed, underexposed, static image, still frame, "
        "flicker, jitter, distorted product, deformed object, duplicate object, extra object, "
        "bad geometry, warped edges, text, subtitles, watermark, logo error, messy background"
    )

    generator = torch.Generator(device="cuda").manual_seed(
        int.from_bytes(os.urandom(4), "big")
    )

    log(
        f"Generating Wan video: {width}x{height}, frames={num_frames}, "
        f"fps={WAN_FPS}, steps={steps}, guidance={WAN_GUIDANCE_SCALE}, quality={quality}"
    )

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=WAN_GUIDANCE_SCALE,
            num_inference_steps=steps,
            generator=generator,
        )

    export_to_video(result.frames[0], output_path, fps=WAN_FPS)

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
    output_path = os.path.join(work_dir, "final.mp4")

    try:
        log(f"Job started: {job_id}")
        log(f"Downloading image: {image_url}")

        download_file(image_url, raw_image_path)

        pipe = get_pipeline()
        image = prepare_image(raw_image_path, pipe, aspect_ratio, quality)

        generate_video(
            prompt=prompt,
            image=image,
            output_path=output_path,
            duration=duration,
            quality=quality,
        )

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
            "worker_version": WORKER_VERSION,
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
        log(f"Job failed: {job_id}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "worker_version": WORKER_VERSION}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


log("BOOT: starting RunPod serverless worker")
runpod.serverless.start({"handler": handler})
