"""Check a running server with synthetic images; requires Pillow, no model download."""

import argparse
import io
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image


MODEL_ID = "fancyfeast/joytag"
MODEL_REVISION = "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
MODEL_SHA256 = "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097"
IMAGE_SIZE = (96, 32)
TOP_K = 15


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def request_json(url, *, body=None, headers=None, timeout=120):
    request = urllib.request.Request(url, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.load(error)


def wait_for_health(url, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    last_error = "server did not become ready"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"startup timeout after {wait_seconds:g}s: {last_error}")
        try:
            status, health = request_json(url + "/health", timeout=min(5, remaining))
            if status == 200 and isinstance(health, dict) and health.get("loaded") is True:
                return health
            last_error = f"/health returned {status}: {health}"
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


def multipart(parts):
    """parts is [(field name, filename, bytes), ...]."""
    boundary = "joytag-smoke-" + uuid.uuid4().hex
    chunks = []
    for field, filename, contents in parts:
        chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                       f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n').encode("ascii"))
        chunks.extend((contents, b"\r\n"))
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def post_images(url, parts, query="?threshold=0.4&top_k=15"):
    body, headers = multipart(parts)
    return request_json(url + "/analyze" + query, body=body, headers=headers)


def validate_predictions(predictions, *, label):
    require(isinstance(predictions, list), f"{label} must be an array")
    names = set()
    previous = math.inf
    for prediction in predictions:
        require(isinstance(prediction, dict), f"{label} contains an invalid entry")
        tag, score = prediction.get("tag"), prediction.get("score")
        require(isinstance(tag, str) and bool(tag.strip()) and tag not in names,
                f"{label} contains an empty or repeated tag")
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                f"{label} contains a nonfinite or out-of-range probability")
        require(score <= previous, f"{label} must be sorted by decreasing probability")
        names.add(tag)
        previous = score


def validate_response(data, filenames, *, threshold=0.4):
    require(data.get("model_id") == MODEL_ID and data.get("device") == "cpu", "unexpected analysis model/device")
    require(data.get("top_k") == TOP_K and data.get("threshold") == threshold, "analysis settings were not preserved")
    results = data.get("results")
    require(isinstance(results, list) and len(results) == len(filenames), "analysis dropped an uploaded image")
    for result, filename in zip(results, filenames):
        require(result.get("filename") == filename, "upload order or filename was not preserved")
        require((result.get("width"), result.get("height")) == IMAGE_SIZE, "incorrect source image dimensions")
        top_tags = result.get("top_tags")
        validate_predictions(top_tags, label="top_tags")
        require(len(top_tags) == TOP_K, "top_tags must contain exactly 15 predictions")
        tags = result.get("tags")
        validate_predictions(tags, label="tags")
        require(result.get("tag_count") == len(tags), "tag_count does not match tags")
        require(all(tag["score"] > threshold for tag in tags), "threshold filtering is incorrect")
        for key in ("preprocess_sec", "inference_sec", "postprocess_sec", "analysis_sec"):
            value = result.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid {key}")
    return results


def run(url, wait_seconds):
    health = wait_for_health(url, wait_seconds)
    require(health.get("model_id") == MODEL_ID, "unexpected model ID")
    require(health.get("device") == "cpu" and health.get("providers") == ["CPUExecutionProvider"], "server must use CPU inference")
    require(health.get("tag_count") == 5813 and health.get("image_size") == 448, "unexpected JoyTag model dimensions")
    require(health.get("output_kind") == "logits" and health.get("output_terminal_op") == "Gemm", "unexpected ONNX output semantics")
    manifest = health.get("model_manifest")
    require(isinstance(manifest, dict), "model manifest is absent from health")
    require(manifest.get("model_id") == MODEL_ID and manifest.get("revision") == MODEL_REVISION, "model revision is not pinned correctly")
    artifacts = manifest.get("files", {})
    require(isinstance(artifacts, dict), "model manifest artifacts must be keyed by filename")
    require(artifacts.get("model.onnx", {}).get("sha256") == MODEL_SHA256, "incorrect model SHA-256")
    require(health.get("max_files") == 16, "expected a 16-image request limit")

    image = io.BytesIO()
    Image.new("RGB", IMAGE_SIZE, (80, 140, 200)).save(image, format="PNG")
    png = image.getvalue()
    filenames = ["synthetic-a.png", "synthetic-b.png"]
    status, data = post_images(url, [("files", name, png) for name in filenames])
    require(status == 200, f"valid multipart upload returned {status}: {data}")
    results = validate_response(data, filenames)
    require(results[0]["top_tags"] == results[1]["top_tags"], "repeated image inference is not deterministic")
    require(results[0]["tags"] == results[1]["tags"], "repeated image thresholded tags differ")

    status, single = post_images(url, [("image", "single.png", png)], "?threshold=1&top_k=15")
    require(status == 200, f"image alias returned {status}")
    single_result = validate_response(single, ["single.png"], threshold=1)[0]
    require(single_result["top_tags"] == results[0]["top_tags"], "top tags changed across requests or threshold settings")
    require(single_result["tags"] == [], "threshold 1 must yield no thresholded tags")

    status, aliases = post_images(url, [("images", "alias.png", png)])
    require(status == 200, f"images alias returned {status}")
    alias_result = validate_response(aliases, ["alias.png"])[0]
    require(alias_result["top_tags"] == results[0]["top_tags"], "images alias altered predictions")

    error_checks = [
        ("invalid_image", [("files", "invalid.png", b"not an image")], "", 415),
        ("empty_image", [("files", "empty.png", b"")], "", 422),
        ("threshold_high", [("files", "range.png", png)], "?threshold=1.5", 422),
        ("threshold_low", [("files", "range.png", png)], "?threshold=-0.1", 422),
        ("top_k_low", [("files", "range.png", png)], "?top_k=0", 422),
        ("top_k_high", [("files", "range.png", png)], "?top_k=501", 422),
        ("too_many_images", [("files", f"limit-{index}.png", png) for index in range(17)], "", 413),
    ]
    for name, parts, query, expected_status in error_checks:
        status, error = post_images(url, parts, query)
        require(status == expected_status, f"{name}: expected {expected_status}, received {status}: {error}")
    status, error = request_json(url + "/analyze", body=b"")
    require(status == 422, f"missing_image: expected 422, received {status}: {error}")
    return {"passed": True, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "checks": ["cpu_metadata", "model_lock", "multipart_files", "15_sorted_finite_tags",
                       "determinism", "image_alias", "images_alias", "threshold_independent_top_tags",
                       "missing_image"] + [item[0] for item in error_checks]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="running server base URL")
    parser.add_argument("--wait-seconds", type=float, default=120, help="maximum startup wait (default: 120)")
    args = parser.parse_args()
    parsed = urllib.parse.urlsplit(args.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        parser.error("--url must be HTTP(S) and --wait-seconds must be a positive finite number")
    try:
        report = run(args.url.rstrip("/"), args.wait_seconds)
    except Exception as error:
        print(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, separators=(",", ":")))
        return 1
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
