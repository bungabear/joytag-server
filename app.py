"""Single-process JoyTag ONNX CPU image tagging server.

The Docker image includes pinned model artifacts in MODEL_DIR. The running
service loads those artifacts locally and never downloads weights.
Preprocessing follows https://github.com/fpgaminer/joytag exactly: white square
padding, Pillow bicubic resize, RGB float32 / 255, and CLIP normalization.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import gc
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Annotated
import warnings

import fastapi
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
import numpy as np
import onnx
import onnxruntime as ort
import PIL
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


MODEL_ID = "fancyfeast/joytag"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
MAX_FILES = 16
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_DIMENSION = 8192
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def env_threads(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not 1 <= value <= 256:
        raise ValueError(f"{name} must be between 1 and 256")
    return value


def inspect_output_kind(path: Path, output_name: str) -> tuple[str, str]:
    """Check the graph's terminal activation instead of guessing from values."""
    model = onnx.load_model(path, load_external_data=False)
    producers = {name: node for node in model.graph.node for name in node.output}
    node = producers.get(output_name)
    passthrough = {"Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze", "Flatten"}
    seen: set[str] = set()
    while node is not None and node.op_type in passthrough and node.input:
        if node.input[0] in seen:
            raise ValueError("Cycle while inspecting ONNX output")
        seen.add(node.input[0])
        node = producers.get(node.input[0])
    terminal_op = node.op_type if node is not None else "unknown"
    requested = os.environ.get("MODEL_OUTPUT_KIND", "auto").lower()
    if requested not in {"auto", "logits", "probabilities"}:
        raise ValueError("MODEL_OUTPUT_KIND must be auto, logits, or probabilities")
    if requested == "auto":
        if terminal_op == "Sigmoid":
            kind = "probabilities"
        elif terminal_op in {"Gemm", "MatMul", "Add", "ReduceMean"}:
            kind = "logits"
        else:
            raise ValueError(
                f"Unexpected terminal output op {terminal_op!r}; inspect the model "
                "and explicitly set MODEL_OUTPUT_KIND"
            )
    else:
        kind = requested
    del model, producers, node
    gc.collect()
    return kind, terminal_op


class JoyTagRuntime:
    def __init__(self) -> None:
        started = time.perf_counter()
        model_path = MODEL_DIR / "model.onnx"
        config_path = MODEL_DIR / "config.json"
        tags_path = MODEL_DIR / "top_tags.txt"
        for path in (model_path, config_path, tags_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required cached model artifact missing: {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.image_size = int(config["image_size"])
        self.tags = [tag.strip() for tag in tags_path.read_text(encoding="utf-8").splitlines() if tag.strip()]
        if self.image_size != 448 or len(self.tags) != int(config["n_tags"]):
            raise ValueError("Unexpected JoyTag image size or tag count")

        self.intra_threads = env_threads("ORT_INTRA_THREADS", 4)
        self.inter_threads = env_threads("ORT_INTER_THREADS", 1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_threads
        options.inter_op_num_threads = self.inter_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one JoyTag ONNX input and one output")
        self.input = inputs[0]
        self.output = outputs[0]
        shape = self.input.shape
        if len(shape) != 4 or self.input.type != "tensor(float)":
            raise ValueError(f"Expected rank-4 FP32 image input, got {shape}, {self.input.type}")
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("This server requires fixed batch 1 or a dynamic batch axis")
        if shape[1] == 3:
            self.layout = "NCHW"
            spatial = shape[2:4]
        elif shape[3] == 3:
            self.layout = "NHWC"
            spatial = shape[1:3]
        else:
            raise ValueError(f"Cannot identify RGB channel axis in {shape}")
        if any(isinstance(dim, int) and dim != self.image_size for dim in spatial):
            raise ValueError(f"ONNX spatial dimensions do not match config: {shape}")
        if len(self.output.shape) != 2:
            raise ValueError(f"Expected rank-2 tag output, got {self.output.shape}")
        tag_dim = self.output.shape[1]
        if isinstance(tag_dim, int) and tag_dim != len(self.tags):
            raise ValueError("ONNX output and tag vocabulary sizes differ")
        self.output_kind, terminal_op = inspect_output_kind(model_path, self.output.name)
        manifest_path = MODEL_DIR / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else None
        )
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        try:
            ffprobe_version = subprocess.run(
                ["ffprobe", "-version"], check=True, capture_output=True,
                text=True, timeout=5
            ).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            ffprobe_version = None
        self.metadata = {
            "loaded": True,
            "model_id": MODEL_ID,
            "device": "cpu",
            "providers": self.session.get_providers(),
            "model_path": str(model_path),
            "model_bytes": model_path.stat().st_size,
            "model_manifest": manifest,
            "model_load_sec": time.perf_counter() - started,
            "input_name": self.input.name,
            "input_shape": shape,
            "input_dtype": self.input.type,
            "input_layout": self.layout,
            "output_name": self.output.name,
            "output_shape": self.output.shape,
            "output_kind": self.output_kind,
            "output_terminal_op": terminal_op,
            "image_size": self.image_size,
            "tag_count": len(self.tags),
            "intra_threads": self.intra_threads,
            "inter_threads": self.inter_threads,
            "execution_mode": "sequential",
            "workers": 1,
            "os_visible_cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "runtime_versions": {
                "onnxruntime": ort.__version__, "onnx": onnx.__version__,
                "numpy": np.__version__, "pillow": PIL.__version__,
                "fastapi": fastapi.__version__,
            },
            "ffprobe_version": ffprobe_version,
            "max_files": MAX_FILES,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_image_dimension": MAX_IMAGE_DIMENSION,
        }

    def preprocess(self, data: bytes) -> tuple[np.ndarray, int, int]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if (
                    width * height > MAX_IMAGE_PIXELS
                    or max(width, height) > MAX_IMAGE_DIMENSION
                ):
                    raise HTTPException(status_code=413, detail="Decoded image exceeds size limits")
                image.load()
                rgb = image.convert("RGB")
        max_dim = max(width, height)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(rgb, ((max_dim - width) // 2, (max_dim - height) // 2))
        resized = padded.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
        normalized = (pixels - MEAN) / STD
        if self.layout == "NCHW":
            normalized = normalized.transpose(2, 0, 1)
        return np.ascontiguousarray(normalized[np.newaxis], dtype=np.float32), width, height

    def analyze(self, data: bytes, filename: str, threshold: float, top_k: int) -> dict:
        preprocess_started = time.perf_counter()
        try:
            image_tensor, width, height = self.preprocess(data)
        except HTTPException:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(status_code=413, detail="Decoded image exceeds size limits") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=415, detail="Upload is not a supported, decodable image") from exc
        preprocess_sec = time.perf_counter() - preprocess_started

        inference_started = time.perf_counter()
        output = self.session.run([self.output.name], {self.input.name: image_tensor})[0]
        inference_sec = time.perf_counter() - inference_started

        postprocess_started = time.perf_counter()
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape != (1, len(self.tags)) or not np.isfinite(scores).all():
            raise RuntimeError(f"Unexpected model output shape or values: {scores.shape}")
        scores = scores[0]
        if self.output_kind == "logits":
            # A stable sigmoid without overflow for very negative logits.
            probability = np.empty_like(scores)
            nonnegative = scores >= 0
            probability[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
            exp_negative = np.exp(scores[~nonnegative])
            probability[~nonnegative] = exp_negative / (1.0 + exp_negative)
        else:
            if np.any((scores < 0) | (scores > 1)):
                raise RuntimeError("Probability output contains values outside [0, 1]")
            probability = scores
        order = np.argsort(-probability, kind="stable")
        selected = [index for index in order if probability[index] > threshold]
        def tagged(index: int) -> dict:
            return {"tag": self.tags[index], "score": float(probability[index])}
        tags = [tagged(index) for index in selected]
        top_tags = [tagged(index) for index in order[:top_k]]
        postprocess_sec = time.perf_counter() - postprocess_started
        return {
            "filename": filename,
            "width": width,
            "height": height,
            "preprocess_sec": preprocess_sec,
            "inference_sec": inference_sec,
            "postprocess_sec": postprocess_sec,
            "analysis_sec": preprocess_sec + inference_sec + postprocess_sec,
            "tag_count": len(tags),
            "tags": tags,
            "top_tags": top_tags,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = await run_in_threadpool(JoyTagRuntime)
    # Serialize inference requests to bound CPU inference concurrency while
    # retaining the configured ONNX thread count.
    app.state.analysis_lock = asyncio.Lock()
    yield
    del app.state.runtime


app = FastAPI(title="JoyTag CPU Server", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict:
    return request.app.state.runtime.metadata


@app.post("/analyze")
async def analyze(
    request: Request,
    files: Annotated[list[UploadFile] | None, File()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.4,
    top_k: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict:
    started = time.perf_counter()
    uploads = list(files or []) + list(images or [])
    if image is not None:
        uploads.append(image)
    if not uploads:
        raise HTTPException(status_code=422, detail="Upload an image using files, image, or images")
    try:
        if len(uploads) > MAX_FILES:
            raise HTTPException(status_code=413, detail=f"At most {MAX_FILES} images per request")
        results = []
        async with request.app.state.analysis_lock:
            for upload in uploads:
                data = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded image exceeds 20 MiB")
                if not data:
                    raise HTTPException(status_code=422, detail="Uploaded image is empty")
                result = await run_in_threadpool(
                    request.app.state.runtime.analyze, data,
                    (upload.filename or "image")[:256], threshold, top_k,
                )
                results.append(result)
        return {
            "model_id": MODEL_ID,
            "device": "cpu",
            "threshold": threshold,
            "top_k": top_k,
            "results": results,
            "request_sec": time.perf_counter() - started,
        }
    finally:
        for upload in uploads:
            await upload.close()
