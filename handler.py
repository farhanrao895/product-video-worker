import os
import time
import uuid
import requests
import runpod
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = os.environ.get("MODEL_REPO", "Lightricks/LTX-Video")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/huggingface-cache")


def download_file(url: str, output_path: str) -> str:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt")
    image_url = job_input.get("image_url")
    duration = job_input.get("duration", 5)
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    quality = job_input.get("quality", "standard")

    if not prompt:
        return {"success": False, "error": "Missing prompt"}

    if not image_url:
        return {"success": False, "error": "Missing image_url"}

    work_dir = f"/tmp/job-{uuid.uuid4()}"
    os.makedirs(work_dir, exist_ok=True)

    image_path = os.path.join(work_dir, "input.png")

    try:
        download_file(image_url, image_path)

        model_path = snapshot_download(
            repo_id=MODEL_REPO,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN
        )

        time.sleep(3)

        return {
            "success": True,
            "message": "RunPod GitHub worker is working. Hugging Face model is accessible.",
            "received": {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "quality": quality
            },
            "model_repo": MODEL_REPO,
            "model_path": model_path,
            "video_url": None
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


runpod.serverless.start({"handler": handler})
