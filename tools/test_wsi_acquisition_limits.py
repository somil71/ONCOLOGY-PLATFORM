"""WSI resource bounds and streaming payload guards using synthetic sources."""

import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from tools.extract_kirc_wsi_features import check_download_budget
from tools.download_primary_rna import verified_download


class Response(io.BytesIO):
    status = 200
    headers = {}


def main():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder / "existing.svs").write_bytes(b"1234")
        (folder / "candidate.partial").write_bytes(b"123")
        (folder / "other.partial").write_bytes(b"12")
        measured = check_download_budget(folder, "candidate", 8, 14, 5, free_bytes=10)
        assert (
            measured["cached_source_and_partial_bytes"] == 9
            and measured["expected_additional_bytes"] == 5
        )
        for budget, reserve in [(13, 5), (14, 6)]:
            try:
                check_download_budget(
                    folder, "candidate", 8, budget, reserve, free_bytes=10
                )
            except ValueError:
                pass
            else:
                raise AssertionError("A raw-cache or free-disk limit was ignored")
        payload = b"abcdefgh"
        item = {
            "Patient_ID": "TEST",
            "file_id": "safe",
            "file_size": len(payload),
            "md5sum": hashlib.md5(payload).hexdigest(),
        }
        with (
            patch(
                "tools.download_primary_rna.urlopen",
                side_effect=lambda *a, **k: Response(payload),
            ),
            patch("tools.download_primary_rna.time.sleep"),
        ):
            result = verified_download(item, folder, suffix=".svs")
            assert (
                result["status"] == "verified_download"
                and (folder / "safe.svs").read_bytes() == payload
            )
        oversized = {
            **item,
            "file_id": "oversized",
            "file_size": 4,
            "md5sum": hashlib.md5(payload[:4]).hexdigest(),
        }
        with (
            patch(
                "tools.download_primary_rna.urlopen",
                side_effect=lambda *a, **k: Response(payload),
            ),
            patch("tools.download_primary_rna.time.sleep"),
        ):
            result = verified_download(oversized, folder, suffix=".svs")
            assert (
                result["status"] == "failed"
                and "exceeds" in result["error"]
                and not (folder / "oversized.svs").exists()
            )
        limited = {**item, "file_id": "reserve"}
        with (
            patch(
                "tools.download_primary_rna.urlopen",
                side_effect=lambda *a, **k: Response(payload),
            ),
            patch("tools.download_primary_rna.time.sleep"),
            patch(
                "tools.download_primary_rna.shutil.disk_usage",
                return_value=SimpleNamespace(free=7),
            ),
        ):
            result = verified_download(
                limited, folder, suffix=".svs", minimum_free_bytes=4
            )
            assert (
                result["status"] == "failed"
                and "reserve" in result["error"]
                and not (folder / "reserve.svs").exists()
            )
    print(
        "PASS: partial-aware raw-cache bound, free-disk reserve, verified streaming download, oversized-payload rejection and per-chunk reserve guard."
    )


if __name__ == "__main__":
    main()
