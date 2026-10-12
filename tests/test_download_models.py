"""Model cache and integrity checks without downloading real model weights."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_models


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def assert_no_partial_files(self):
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_valid_cache_is_verified_and_reused_without_network(self):
        contents = b"verified cached model fixture"
        destination = self.directory / "model.onnx"
        destination.write_bytes(contents)
        with mock.patch.object(download_models, "urlopen") as network:
            result = download_models.download_artifact("model.onnx", self.directory, digest(contents))
        network.assert_not_called()
        self.assertEqual(destination.read_bytes(), contents)
        self.assertEqual(result["sha256"], digest(contents))
        self.assertEqual(result["bytes"], len(contents))
        self.assertEqual(result["url"],
                         f"https://huggingface.co/{download_models.MODEL_ID}/resolve/{download_models.REVISION}/model.onnx")
        self.assert_no_partial_files()

    def test_corrupt_cache_is_replaced_only_after_streamed_download_verification(self):
        old_contents = b"incomplete cached model"
        new_contents = bytes(range(256)) * 3
        destination = self.directory / "model.onnx"
        destination.write_bytes(old_contents)
        self_test = self

        class Response(io.BytesIO):
            def read(self, size=-1):
                self_test.assertEqual(destination.read_bytes(), old_contents,
                                      "the existing file must survive until the download is complete and verified")
                return super().read(size)

        with mock.patch.object(download_models, "CHUNK_BYTES", 37), \
                mock.patch.object(download_models, "urlopen", return_value=Response(new_contents)) as network:
            result = download_models.download_artifact("model.onnx", self.directory, digest(new_contents))
        network.assert_called_once()
        request = network.call_args.args[0]
        self.assertEqual(request.full_url,
                         f"https://huggingface.co/{download_models.MODEL_ID}/resolve/{download_models.REVISION}/model.onnx")
        self.assertEqual(network.call_args.kwargs["timeout"], download_models.DOWNLOAD_TIMEOUT_SEC)
        self.assertEqual(destination.read_bytes(), new_contents)
        self.assertEqual(download_models.sha256_file(destination), digest(new_contents))
        self.assertEqual(result["sha256"], digest(new_contents))
        self.assertEqual(result["bytes"], len(new_contents))
        self.assert_no_partial_files()

    def test_hash_mismatch_never_installs_download_or_destroys_existing_file(self):
        destination = self.directory / "model.onnx"
        for existing in (None, b"previous cache"):
            with self.subTest(existing=existing):
                if existing is not None:
                    destination.write_bytes(existing)
                with mock.patch.object(download_models, "urlopen", return_value=io.BytesIO(b"wrong bytes")):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        download_models.download_artifact("model.onnx", self.directory, digest(b"expected bytes"))
                if existing is None:
                    self.assertFalse(destination.exists())
                else:
                    self.assertEqual(destination.read_bytes(), existing)
                self.assert_no_partial_files()

    def test_interrupted_download_preserves_cache_and_removes_partial_file(self):
        destination = self.directory / "model.onnx"
        destination.write_bytes(b"old cache")

        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise URLError("fixture connection interrupted")
                return super().read(size)

        with mock.patch.object(download_models, "urlopen", return_value=InterruptedResponse(b"first chunk")):
            with self.assertRaises(URLError):
                download_models.download_artifact("model.onnx", self.directory, digest(b"complete model"))
        self.assertEqual(destination.read_bytes(), b"old cache")
        self.assert_no_partial_files()

    def test_manifest_uses_pinned_lock_reuses_cache_and_survives_failed_refresh(self):
        self.assertEqual(download_models.MODEL_ID, "fancyfeast/joytag")
        self.assertEqual(download_models.REVISION, "6b7f16331a6ccf0fdce37d5a9564715f6e772b22")
        self.assertEqual(download_models.ARTIFACTS, {
            "config.json": "b53ee29a8f0d6353331f455fd640c70ef45e5bd979b656f7b9983cffa3724562",
            "top_tags.txt": "32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0",
            "model.onnx": "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097",
        })
        contents = {"config.json": b'{"image_size": 448}',
                    "top_tags.txt": b"tag_one\ntag_two\n", "model.onnx": b"synthetic model fixture"}
        hashes = {name: digest(data) for name, data in contents.items()}

        def response(request, timeout):
            filename = request.full_url.rsplit("/", 1)[1]
            self.assertIn(f"/resolve/{download_models.REVISION}/", request.full_url)
            return io.BytesIO(contents[filename])

        with mock.patch.object(download_models, "ARTIFACTS", hashes):
            with mock.patch.object(download_models, "urlopen", side_effect=response) as network:
                manifest = download_models.download_models(self.directory)
            self.assertEqual(network.call_count, len(contents))
            self.assertEqual(manifest["model_id"], download_models.MODEL_ID)
            self.assertEqual(manifest["revision"], download_models.REVISION)
            self.assertEqual(set(manifest["files"]), set(contents))
            for filename, data in contents.items():
                self.assertEqual((self.directory / filename).read_bytes(), data)
                self.assertEqual(manifest["files"][filename]["sha256"], digest(data))
                self.assertEqual(manifest["files"][filename]["bytes"], len(data))
            manifest_path = self.directory / "manifest.json"
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
            with mock.patch.object(download_models, "urlopen") as network:
                self.assertEqual(download_models.download_models(self.directory), manifest)
            network.assert_not_called()

            original_manifest = manifest_path.read_bytes()
            (self.directory / "top_tags.txt").write_bytes(b"corrupt cache")
            with mock.patch.object(download_models, "urlopen", return_value=io.BytesIO(b"wrong replacement")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    download_models.download_models(self.directory)
            self.assertEqual(manifest_path.read_bytes(), original_manifest,
                             "a failed refresh must not publish an unverified manifest")
            self.assert_no_partial_files()


if __name__ == "__main__":
    unittest.main()
