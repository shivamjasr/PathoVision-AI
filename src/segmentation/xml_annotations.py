from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


@dataclass(frozen=True)
class Polygon:
    points: np.ndarray  # shape [N, 2], level-0 x/y coordinates


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def load_annotation_polygons(xml_path: str | Path):
    """Parse common CAMELYON/ASAP-style contour XML.

    The parser searches for Annotation -> Coordinates -> Coordinate
    elements without depending on XML namespaces.
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Annotation XML not found: {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    for annotation in root.iter():
        if _local_name(annotation.tag) != "annotation":
            continue

        coordinates = None
        for child in annotation.iter():
            if _local_name(child.tag) == "coordinates":
                coordinates = child
                break

        if coordinates is None:
            continue

        points = []
        for coordinate in coordinates:
            if _local_name(coordinate.tag) != "coordinate":
                continue

            x = coordinate.attrib.get("X")
            y = coordinate.attrib.get("Y")

            if x is None or y is None:
                x = coordinate.attrib.get("x")
                y = coordinate.attrib.get("y")

            if x is None or y is None:
                continue

            points.append(
                [float(x), float(y)]
            )

        if len(points) >= 3:
            polygons.append(
                Polygon(
                    points=np.asarray(
                        points,
                        dtype=np.float64,
                    )
                )
            )

    if not polygons:
        raise ValueError(
            f"No usable polygons found in {xml_path}. "
            "Check that the annotation format contains "
            "Annotation/Coordinates/Coordinate elements."
        )

    return polygons


def render_patch_mask(
    polygons,
    x: int,
    y: int,
    width: int,
    height: int,
    level_downsample: float = 1.0,
) -> np.ndarray:
    """Rasterize level-0 polygons into a local patch mask.

    Parameters
    ----------
    polygons:
        Annotation polygons in level-0 coordinates.
    x, y:
        Level-0 top-left patch coordinate.
    width, height:
        Patch dimensions in pixels at the target level.
    level_downsample:
        Convert level-0 annotation coordinates to target level.
        Keep 1.0 when the patch itself is read at level 0.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Patch dimensions must be positive.")

    if level_downsample <= 0:
        raise ValueError("level_downsample must be positive.")

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    x = float(x)
    y = float(y)

    for polygon in polygons:
        points = polygon.points.copy()

        # Convert from level-0 coordinates to local target-level coords.
        points[:, 0] = points[:, 0] / level_downsample - x
        points[:, 1] = points[:, 1] / level_downsample - y

        polygon_points = np.round(points).astype(np.int32)

        cv2.fillPoly(
            mask,
            [polygon_points],
            255,
        )

    return mask


def tumor_fraction(mask: np.ndarray) -> float:
    """Return the fraction of pixels marked as tumor."""
    if mask.size == 0:
        return 0.0
    return float((mask > 0).mean())
