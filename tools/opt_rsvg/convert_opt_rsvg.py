#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert_opt_rsvg.py

Convert OPT-RSVG (visual grounding in remote-sensing images) to a unified
JSONL intermediate format used by this repo's RS reproduction docs.

Dataset notes (as packaged by OPT-RSVG):
- Images: `Image.zip` containing `Image/<id>.jpg`
- Annotations: `Annotations.zip` containing `Annotations/<id>.xml`
- Splits: `split.zip` containing `split/{train,val,test}.txt` with global
  object indices (0..48951). Each `<object>` entry in each XML is a sample.

This script supports reading annotations/splits directly from the zip files.

Typical usage:

  python tools/opt_rsvg/convert_opt_rsvg.py \
    --root datasets/opt_rsvg/OPT-RSVG \
    --out-dir datasets/opt_rsvg/processed

Outputs:
  - opt_rsvg_train.jsonl / opt_rsvg_val.jsonl / opt_rsvg_test.jsonl
  - meta_stats.json
  - bad_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it: Iterable, **_: Any):
        return it


_INT_LINE_RE = re.compile(r"^\s*(\d+)\s*$")


@dataclass
class XmlObject:
    name: str
    bbox_xyxy: list[int]
    description: str


@dataclass
class XmlAnn:
    filename: str
    width: int
    height: int
    objects: list[XmlObject]


def _parse_xml(xml_bytes: bytes) -> XmlAnn:
    root = ET.fromstring(xml_bytes)

    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    if size is None:
        raise ValueError("Missing <size> in xml")
    width = int(float(size.findtext("width") or 0))
    height = int(float(size.findtext("height") or 0))

    objects: list[XmlObject] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = int(float(box.findtext("xmin") or 0))
        ymin = int(float(box.findtext("ymin") or 0))
        xmax = int(float(box.findtext("xmax") or 0))
        ymax = int(float(box.findtext("ymax") or 0))
        description = (obj.findtext("description") or "").strip()
        objects.append(XmlObject(name=name, bbox_xyxy=[xmin, ymin, xmax, ymax], description=description))

    return XmlAnn(filename=filename, width=width, height=height, objects=objects)


def _bbox_to_polygon(bbox_xyxy: list[int]) -> list[list[int]]:
    xmin, ymin, xmax, ymax = bbox_xyxy
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def _load_split_ids(root: Path, split: str) -> set[int]:
    split_txt = root / "split" / f"{split}.txt"
    if split_txt.exists():
        lines = split_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        split_zip = root / "split.zip"
        if not split_zip.exists():
            raise FileNotFoundError(f"Cannot find split files under: {root}")
        with zipfile.ZipFile(split_zip) as zf:
            lines = zf.read(f"split/{split}.txt").decode("utf-8", errors="ignore").splitlines()

    ids: set[int] = set()
    for line in lines:
        m = _INT_LINE_RE.match(line)
        if not m:
            continue
        ids.add(int(m.group(1)))
    return ids


def _iter_xml_entries(root: Path) -> tuple[list[str], "XmlReader"]:
    ann_dir = root / "Annotations"
    ann_zip = root / "Annotations.zip"

    if ann_dir.is_dir():
        xml_files = sorted([p.name for p in ann_dir.glob("*.xml")])

        def reader(name: str) -> bytes:
            return (ann_dir / name).read_bytes()

        return xml_files, reader

    if ann_zip.exists():
        zf = zipfile.ZipFile(ann_zip)
        names = []
        for n in zf.namelist():
            if not n.lower().endswith(".xml"):
                continue
            if not n.startswith("Annotations/"):
                continue
            if "__MACOSX" in n or "/._" in n:
                continue
            names.append(n)

        # Sort by basename to match split indexing convention.
        names = sorted(names, key=lambda x: os.path.basename(x))

        def reader(name: str) -> bytes:
            return zf.read(name)

        return names, reader

    raise FileNotFoundError(f"Cannot find Annotations/ or Annotations.zip under: {root}")


class XmlReader:
    def __call__(self, name: str) -> bytes: ...


def _open_writers(out_dir: Path) -> dict[str, IO[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (out_dir / "opt_rsvg_train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "opt_rsvg_val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "opt_rsvg_test.jsonl").open("w", encoding="utf-8"),
    }


def convert_opt_rsvg(root: Path, out_dir: Path, *, img_root: Path | None = None) -> dict[str, Any]:
    train_ids = _load_split_ids(root, "train")
    val_ids = _load_split_ids(root, "val")
    test_ids = _load_split_ids(root, "test")
    all_ids = train_ids | val_ids | test_ids
    expected_total = len(all_ids)

    if expected_total == 0:
        raise ValueError("Empty split ids. Bad split files?")
    if min(all_ids) != 0 or max(all_ids) != expected_total - 1:
        raise ValueError(
            f"Split ids are not a contiguous 0..N-1 range: min={min(all_ids)}, max={max(all_ids)}, N={expected_total}"
        )

    if img_root is None:
        if (root / "Image").is_dir():
            img_root = root / "Image"
        elif (root / "JPEGImages").is_dir():
            img_root = root / "JPEGImages"
        else:
            img_root = root / "Image"  # default expected extracted folder

    xml_names, xml_reader = _iter_xml_entries(root)
    writers = _open_writers(out_dir)
    bad_f = (out_dir / "bad_samples.jsonl").open("w", encoding="utf-8")

    stats: dict[str, Any] = {
        "total": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "bad": 0,
        "categories": Counter(),
        "expr_len": [],
        "bbox_area": [],
        "image_sizes": Counter(),
    }

    xml_cache: dict[str, XmlAnn] = {}
    global_idx = 0
    for xml_name in tqdm(xml_names, desc="OPT-RSVG xml"):
        # parse xml (with cache)
        cache_key = os.path.basename(xml_name)
        if cache_key not in xml_cache:
            xml_cache[cache_key] = _parse_xml(xml_reader(xml_name))
        ann = xml_cache[cache_key]

        for obj_idx, obj in enumerate(ann.objects):
            if global_idx not in all_ids:
                bad_f.write(
                    json.dumps(
                        {
                            "sample_id": str(global_idx),
                            "reason": "id_not_in_any_split",
                            "xml": cache_key,
                            "obj_idx": obj_idx,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["bad"] += 1
                global_idx += 1
                continue

            expr = obj.description
            if not expr:
                bad_f.write(
                    json.dumps(
                        {
                            "sample_id": str(global_idx),
                            "reason": "empty_expression",
                            "xml": cache_key,
                            "obj_idx": obj_idx,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["bad"] += 1
                global_idx += 1
                continue

            split: str
            if global_idx in train_ids:
                split = "train"
            elif global_idx in val_ids:
                split = "val"
            else:
                split = "test"

            image_path = str(img_root / ann.filename)
            bbox = obj.bbox_xyxy
            sample = {
                "sample_id": str(global_idx),
                "split": split,
                "image_path": image_path,
                "mask_path": None,
                "image_id": Path(ann.filename).stem,
                "expr": expr,
                "img_w": ann.width,
                "img_h": ann.height,
                "raw_box_xyxy": bbox,
                "tight_box_xyxy": bbox,
                "polygons": [_bbox_to_polygon(bbox)],
                "poly_meta": {
                    "cc_policy": "bbox_as_polygon",
                    "nmax": 4,
                    "n_vertices": [4],
                    "has_hole": False,
                    "fidelity_iou": 1.0,
                    "source": "opt-rsvg",
                },
                "category": obj.name,
                "notes": None,
            }

            writers[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats[split] += 1
            stats["total"] += 1
            stats["categories"][obj.name] += 1
            stats["expr_len"].append(len(expr.split()))
            stats["bbox_area"].append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            stats["image_sizes"][(ann.width, ann.height)] += 1
            global_idx += 1

    for f in writers.values():
        f.close()
    bad_f.close()

    if global_idx != expected_total:
        raise AssertionError(f"Enumerated samples {global_idx} != expected {expected_total}")

    # basic stats
    import numpy as np

    expr_lens = np.asarray(stats["expr_len"], dtype=np.int32) if stats["expr_len"] else np.asarray([], dtype=np.int32)
    bbox_areas = (
        np.asarray(stats["bbox_area"], dtype=np.float64) if stats["bbox_area"] else np.asarray([], dtype=np.float64)
    )

    meta_stats: dict[str, Any] = {
        "total_samples": int(stats["total"]),
        "train_samples": int(stats["train"]),
        "val_samples": int(stats["val"]),
        "test_samples": int(stats["test"]),
        "bad_samples": int(stats["bad"]),
        "num_xml": len(xml_cache),
        "num_categories": len(stats["categories"]),
        "categories": dict(stats["categories"]),
        "expr_len_stats": {
            "mean": float(expr_lens.mean()) if expr_lens.size else 0.0,
            "p50": float(np.percentile(expr_lens, 50)) if expr_lens.size else 0.0,
            "p90": float(np.percentile(expr_lens, 90)) if expr_lens.size else 0.0,
            "max": int(expr_lens.max()) if expr_lens.size else 0,
        },
        "bbox_area_stats": {
            "mean": float(bbox_areas.mean()) if bbox_areas.size else 0.0,
            "min": float(bbox_areas.min()) if bbox_areas.size else 0.0,
            "max": float(bbox_areas.max()) if bbox_areas.size else 0.0,
        },
        "image_size_top": [
            {"w": int(w), "h": int(h), "count": int(c)}
            for (w, h), c in stats["image_sizes"].most_common(10)
        ],
        "has_mask": False,
        "polygon_source": "bbox_as_rectangle",
    }

    with (out_dir / "meta_stats.json").open("w", encoding="utf-8") as f:
        json.dump(meta_stats, f, indent=2, ensure_ascii=False)

    return meta_stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert OPT-RSVG to JSONL")
    ap.add_argument("--root", required=True, type=Path, help="OPT-RSVG root dir (contains Annotations.zip, split.zip)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    ap.add_argument(
        "--img-root",
        default=None,
        type=Path,
        help="Image directory (default: <root>/Image if exists)",
    )
    args = ap.parse_args()

    meta = convert_opt_rsvg(args.root, args.out_dir, img_root=args.img_root)
    print("\n=== OPT-RSVG conversion complete ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

