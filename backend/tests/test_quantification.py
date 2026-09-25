import numpy as np

from src.segmentation.quantification import quantify_tumor


def test_quantification_counts_tumor_regions():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:7, 2:7] = 255
    mask[12:16, 12:16] = 255
    coverage = np.ones((20, 20), dtype=np.float32)
    tissue = np.ones((20, 20), dtype=bool)

    result = quantify_tumor(mask, coverage, tissue_mask=tissue, region_min_area=3)
    assert result["tumor_area_pixels_thumbnail"] == 25 + 16
    assert result["region_count"] == 2
    assert result["largest_region_pixels_thumbnail"] == 25
