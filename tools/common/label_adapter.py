#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
label_adapter.py - Dataset-agnostic label conversion utilities.

Provides: RLE→mask, mask→bbox, mask→polygon(canonicalized), polygon→mask, IoU.

Polygon canonicalization logic is embedded here (copied from
``data/poly_utils.py``) to avoid triggering the heavy ``data/__init__``
import chain that requires fairseq.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def _decode_coco_rle_counts_string(counts: str) -> list[int]:
    """Decode COCO's compressed RLE counts string.

    This implementation follows the reference algorithm used by COCO's mask
    API (LEB128-style variable-length encoding + a small delta coding trick).
    """

    result: list[int] = []
    p = 0
    m = 0
    while p < len(counts):
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)

        # COCO's RLE uses a delta coding for entries after the first two.
        if m > 2:
            x += result[m - 2]
        result.append(int(x))
        m += 1
    return result


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    """Decode COCO-style RLE into a binary uint8 mask (H, W)."""

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = _decode_coco_rle_counts_string(counts)
    elif isinstance(counts, (list, tuple)):
        counts = [int(x) for x in counts]
    else:
        raise TypeError(f"Unsupported RLE counts type: {type(counts)}")

    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if val == 1:
            flat[idx : idx + c] = 1
        idx += c
        val ^= 1
    if idx != h * w:
        raise ValueError(f"RLE decode length mismatch: {idx} vs {h*w}")

    # COCO uses Fortran order (column-major)
    return flat.reshape((h, w), order="F")


# ---------------------------------------------------------------------------
# Mask / bbox helpers
# ---------------------------------------------------------------------------

def mask_to_box_xyxy(mask: np.ndarray) -> list[int]:
    """Compute tight bbox [x1,y1,x2,y2] from a binary mask.

    Coordinates follow the common *exclusive max* convention used in this
    codebase's IoU computations: width = x2 - x1, height = y2 - y1.
    """

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def polygons_to_mask(polygons: list[list[list[int]]], h: int, w: int) -> np.ndarray:
    """Rasterize polygons to a binary mask (uint8 0/1)."""

    import cv2

    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask


def iou_from_masks(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Polygon canonicalization (mirror of data/poly_utils.py — kept standalone)
# ---------------------------------------------------------------------------

def _is_clockwise(poly: list) -> bool:
    """Shoelace formula: area < 0 ⇒ clockwise.

    ``poly`` is a flat list ``[x0, y0, x1, y1, ...]``.
    (Mirrors ``data/poly_utils.is_clockwise``.)
    """
    n = len(poly) // 2
    xs = list(poly[::2]) + [poly[0]]
    ys = list(poly[1::2]) + [poly[1]]
    area = 0
    for i in range(n):
        area += (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i])
    return area < 0


def _revert_direction(poly: list) -> list:
    """Reverse vertex order (flat list)."""
    arr = np.array(poly).reshape(-1, 2)
    arr = arr[::-1]
    return arr.flatten().tolist()


def _reorder_points(poly: list) -> list:
    """Rotate so that the vertex closest to origin (0,0) comes first."""
    arr = np.array(poly)
    xs = arr[::2]
    ys = arr[1::2]
    pts = arr.reshape(-1, 2)
    start = int(np.argmin(xs ** 2 + ys ** 2))
    reordered = np.concatenate([pts[start:], pts[:start]], axis=0)
    return reordered.flatten().tolist()


def _process_polygons(
    polygons: list[list],
    *,
    redirection: bool = True,
    reorder: bool = True,
    close: bool = False,
) -> list[list]:
    """Canonicalize polygons: clockwise, start closest to origin, sort.

    Mirrors ``data.poly_utils.process_polygons``.
    """
    result = []
    for poly in polygons:
        if redirection and not _is_clockwise(poly):
            poly = _revert_direction(poly)
        if reorder:
            poly = _reorder_points(poly)
        if close:
            pts = np.array(poly).reshape(-1, 2)
            if not np.array_equal(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            poly = pts.flatten().tolist()
        result.append(poly)
    result.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
    return result


# ---------------------------------------------------------------------------
# Mask → Polygon
# ---------------------------------------------------------------------------

@dataclass
class PolygonResult:
    polygons: list[list[list[int]]]
    n_vertices: list[int]
    fidelity_iou: float


def mask_to_polygons(
    mask: np.ndarray,
    *,
    cc_policy: Literal["largest", "multi"] = "largest",
    nmax: int = 80,
    approx_epsilon: float | None = None,
) -> PolygonResult:
    """Convert a binary mask to canonicalized polygons.

    Notes:
    - Uses external contours only (holes are ignored).
    - Canonicalization follows the repo convention (``data/poly_utils``):
      clockwise winding + start vertex closest to origin.
    """

    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return PolygonResult(polygons=[], n_vertices=[], fidelity_iou=0.0)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if cc_policy == "largest":
        contours = contours[:1]

    polygons_flat: list[list[int]] = []
    for contour in contours:
        pts = contour.squeeze(1)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue

        if approx_epsilon is not None and approx_epsilon > 0:
            approx = cv2.approxPolyDP(contour, approx_epsilon, closed=True)
            pts = approx.squeeze(1)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue

        if nmax > 0 and pts.shape[0] > nmax:
            idx = np.linspace(0, pts.shape[0] - 1, num=nmax, dtype=np.int32)
            pts = pts[idx]

        poly_flat: list[int] = []
        for x, y in pts.tolist():
            poly_flat.extend([int(x), int(y)])
        if len(poly_flat) >= 6:
            polygons_flat.append(poly_flat)

    if not polygons_flat:
        return PolygonResult(polygons=[], n_vertices=[], fidelity_iou=0.0)

    # Canonicalize (standalone — no data.__init__ import needed).
    polygons_flat = _process_polygons(
        polygons_flat, redirection=True, reorder=True, close=False
    )

    polygons: list[list[list[int]]] = []
    n_vertices: list[int] = []
    for poly in polygons_flat:
        pts2 = [[int(poly[i]), int(poly[i + 1])] for i in range(0, len(poly), 2)]
        if len(pts2) >= 3:
            polygons.append(pts2)
            n_vertices.append(len(pts2))

    h, w = mask_u8.shape
    recon = polygons_to_mask(polygons, h, w)
    fidelity = iou_from_masks(recon, mask_u8)
    return PolygonResult(polygons=polygons, n_vertices=n_vertices, fidelity_iou=fidelity)
