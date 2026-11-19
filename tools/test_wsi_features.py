"""Physical WSI sampling and original-primary specimen selection regressions."""

import json
from pathlib import Path
from PIL import Image
from src.wsi_features import physical_patch_spec, sample_wsi_bag, tissue_fraction
from tools.prepare_kirc_inventory import choose_primary_diagnostic_slides


class SlideFixture:
    dimensions = (2048, 2048)
    level_downsamples = (1.0, 4.0)
    properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

    def __init__(self, color=(190, 90, 150)):
        self.color = color

    def get_best_level_for_downsample(self, downsample):
        return 1 if downsample >= 4 else 0

    def get_thumbnail(self, size):
        return Image.new("RGB", (256, 256), self.color)

    def read_region(self, position, level, size):
        return Image.new("RGBA", size, (*self.color, 255))


def main():
    slide = SlideFixture()
    spec = physical_patch_spec(slide, 0.5, 224)
    assert spec["level0_patch_extent"] == [448, 448]
    assert spec["read_size_at_level"] == [448, 448]
    images, audit = sample_wsi_bag(slide, "TCGA-TEST", 8, 0.5, 42, min_patches=4)
    images_again, audit_again = sample_wsi_bag(
        slide, "TCGA-TEST", 8, 0.5, 42, min_patches=4
    )
    assert (
        len(images) == 8
        and audit["coordinates_level0"] == audit_again["coordinates_level0"]
    )
    assert all(image.size == (224, 224) for image in images)
    assert all(tissue_fraction(image) == 1 for image in images_again)
    assert tissue_fraction(Image.new("RGB", (32, 32), "white")) == 0
    assert all(
        0 <= x <= 1600 and 0 <= y <= 1600 for x, y in audit["coordinates_level0"]
    )
    missing = SlideFixture()
    missing.properties = {}
    lowresolution = SlideFixture()
    lowresolution.properties = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "1.0"}
    for bad in (missing, lowresolution):
        try:
            physical_patch_spec(bad, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("Unverified/invented physical resolution accepted")
    try:
        sample_wsi_bag(SlideFixture((255, 255, 255)), "TEST", 8, min_patches=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Background-only slide accepted")

    def source(identity, sample_type, patient="TCGA-X"):
        return {
            "file_id": identity,
            "data_type": "Slide Image",
            "experimental_strategy": "Diagnostic Slide",
            "cases": [
                {"submitter_id": patient, "samples": [{"sample_type": sample_type}]}
            ],
        }

    selected = choose_primary_diagnostic_slides(
        [
            source("a", "Additional - New Primary"),
            source("b", "Primary Tumor"),
            source("c", "Primary Tumor"),
            source("d", "Primary Tumor", "OTHER"),
        ],
        {"TCGA-X"},
    )
    assert (
        len(selected) == 1
        and selected[0]["file_id"] == "b"
        and selected[0]["candidate_files"] == 2
    )
    manifest = json.loads(
        Path("data/tcga-kirc/wsi-diagnostic/selected-manifest.json").read_text()
    )
    assert all(
        {
            sample.get("sample_type")
            for case in item["cases"]
            for sample in case.get("samples", [])
        }
        == {"Primary Tumor"}
        for item in manifest["files"]
    )
    actual = next(
        item for item in manifest["files"] if item["Patient_ID"] == "TCGA-DV-A4W0"
    )
    assert actual["file_id"] == "b3c0039f-b969-4187-9fcd-f88b2dbdb84f"
    print(
        "PASS: physical patch extent, deterministic sampling, tissue rejection, missing/anisotropic resolution guards, original-primary WSI selection and actual corrected KIRC manifest."
    )


if __name__ == "__main__":
    main()
