"""Analytic HU/window regression fixtures, not clinical predictions."""

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from src.ct_preprocessing import ct_hounsfield_pixels, ct_window_channels


def main():
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Modality = "CT"
    ds.Rows = 1
    ds.Columns = 3
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np.array([0, 520, 1024], dtype=np.int16).tobytes()
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -1000
    hu, valid = ct_hounsfield_pixels(ds)
    np.testing.assert_allclose(hu, [[-1000, 40, 1048]])
    assert valid.all()
    rgb = ct_window_channels(ds)
    np.testing.assert_allclose(rgb[0, :, 0], [0, 0.5, 1])
    ds.PixelPaddingValue = 0
    rgb = ct_window_channels(ds)
    assert rgb[0, 0].tolist() == [0, 0, 0]
    for field, value in [
        ("Modality", "MR"),
        ("RescaleSlope", 0),
        ("RescaleType", "OTHER"),
        ("RescaleSlope", 1e308),
    ]:
        copy = ds.copy()
        setattr(copy, field, value)
        try:
            ct_hounsfield_pixels(copy)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid CT units/modality accepted")
    del ds.RescaleSlope
    try:
        ct_hounsfield_pixels(ds)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing CT rescale metadata accepted")
    print(
        "PASS: CT HU conversion, fixed windows, padding exclusion and modality/unit validation."
    )


if __name__ == "__main__":
    main()
