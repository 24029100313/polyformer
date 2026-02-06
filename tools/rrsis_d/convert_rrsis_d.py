#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert_rrsis_d.py

Convert RRSIS-D to the unified JSONL intermediate format.

RRSIS-D provides:
- `instances.json` (COCO-like dict with images/annotations/categories)
- `refs(unc).p` (referring expressions with split + ann_id mapping)
- segmentation masks as COCO RLE under `annotations[*].segmentation[0]`

Typical usage:

  python tools/rrsis_d/convert_rrsis_d.py \
    --root datasets/rrsis_d \
    --out-dir datasets/rrsis_d/processed \
    --nmax 80

Outputs:
  - rrsis_d_train.jsonl / rrsis_d_val.jsonl / rrsis_d_test.jsonl
  - meta_stats.json
  - bad_samples.jsonl

Notes:
- By default we do NOT export masks to PNG (to save disk). Enable
  `--export-masks` if you want `mask_path` to exist on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it: Iterable, **_: Any):
        return it


def _find_first(root: Path, candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    cand_str = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Cannot find required file. Tried:\n{cand_str}")


def _default_img_root(root: Path) -> Path:
    candidates = [
        root / "images" / "rrsisd" / "JPEGImages",
        root / "JPEGImages",
        root / "images" / "JPEGImages",
        root / "rrsisd" / "JPEGImages",
    ]
    for c in candidates:
        if c.is_dir():
            return c

    # Fall back to the most common layout used in this repo's notes.
    return candidates[0]


def convert_rrsis_d(
    root: Path,
    out_dir: Path,
    *,
    img_root: Path | None = None,
    nmax: int = 80,
    cc_policy: str = "largest",
    approx_epsilon: float | None = None,
    export_masks: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    # Make `tools.*` and `data.*` imports work even when the script is executed as
    # `python tools/rrsis_d/convert_rrsis_d.py`.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.common.label_adapter import mask_to_box_xyxy, mask_to_polygons, rle_to_mask

    instances_path = _find_first(
        root,
        [
            root / "instances.json",
            root / "rrsisd" / "instances.json",
            root / "refer" / "data" / "rrsisd" / "instances.json",
        ],
    )
    refs_path = _find_first(
        root,
        [
            root / "refs(unc).p",
            root / "rrsisd" / "refs(unc).p",
            root / "refer" / "data" / "rrsisd" / "refs(unc).p",
        ],
    )

    if img_root is None:
        img_root = _default_img_root(root)

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": (out_dir / "rrsis_d_train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "rrsis_d_val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "rrsis_d_test.jsonl").open("w", encoding="utf-8"),
    }
    bad_f = (out_dir / "bad_samples.jsonl").open("w", encoding="utf-8")
    mask_dir = out_dir / "masks"
    if export_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    instances = json.loads(instances_path.read_text(encoding="utf-8", errors="ignore"))
    refs = pickle.loads(refs_path.read_bytes())

    ann_by_id = {a["id"]: a for a in instances.get("annotations", [])}
    img_by_id = {im["id"]: im for im in instances.get("images", [])}
    cat_by_id = {c["id"]: c.get("name") for c in instances.get("categories", [])}

    stats: dict[str, Any] = {
        "total": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "bad": 0,
        "categories": Counter(),
        "expr_len": [],
        "bbox_area": [],
        "fidelity": [],
        "empty_mask": 0,
    }

    # Deterministic ordering helps reproducibility & diff.
    refs = sorted(refs, key=lambda r: int(r.get("ref_id", 0)))
    if limit is not None:
        refs = refs[:limit]

    for ref in tqdm(refs, desc="RRSIS-D refs"):
        ref_id = int(ref["ref_id"])
        ann_id = int(ref["ann_id"])
        image_id = int(ref["image_id"])
        split = ref.get("split")
        if split not in {"train", "val", "test"}:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "bad_split",
                        "split": split,
                        "ann_id": ann_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        ann = ann_by_id.get(ann_id)
        im = img_by_id.get(image_id)
        if ann is None or im is None:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "missing_ann_or_image",
                        "ann_id": ann_id,
                        "image_id": image_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        file_name = ref.get("file_name") or im.get("file_name")
        if not file_name:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "missing_file_name",
                        "image_id": image_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        sent_list = ref.get("sentences") or []
        expr = (sent_list[0].get("sent") if sent_list else "") if isinstance(sent_list, list) else ""
        expr = (expr or "").strip()
        if not expr:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "empty_expression",
                        "ann_id": ann_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        seg_list = ann.get("segmentation")
        if not isinstance(seg_list, list) or not seg_list:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "missing_segmentation",
                        "ann_id": ann_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        rle = seg_list[0]
        try:
            mask = rle_to_mask(rle)
        except Exception as e:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "rle_decode_failed",
                        "ann_id": ann_id,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        if mask.sum() == 0:
            stats["empty_mask"] += 1
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "empty_mask",
                        "ann_id": ann_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        poly_res = mask_to_polygons(mask, cc_policy=cc_policy, nmax=nmax, approx_epsilon=approx_epsilon)
        if not poly_res.polygons:
            bad_f.write(
                json.dumps(
                    {
                        "sample_id": str(ref_id),
                        "reason": "mask_to_polygon_failed",
                        "ann_id": ann_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["bad"] += 1
            continue

        tight_box = mask_to_box_xyxy(mask)
        raw_box = [int(x) for x in ann.get("bbox") or [0, 0, 0, 0]]
        cat_id = int(ann.get("categories_id") or ref.get("category_id") or -1)
        cat_name = cat_by_id.get(cat_id)

        mask_path: str | None
        if export_masks:
            import cv2

            mask_path = str(mask_dir / f"{ref_id}.png")
            cv2.imwrite(mask_path, (mask * 255).astype("uint8"))
        else:
            mask_path = None

        sample = {
            "sample_id": str(ref_id),
            "split": split,
            "image_path": str(img_root / file_name),
            "mask_path": mask_path,
            "image_id": Path(file_name).stem,
            "image_id_int": image_id,
            "ref_id": ref_id,
            "ann_id": ann_id,
            "expr": expr,
            "img_w": int(im.get("width") or mask.shape[1]),
            "img_h": int(im.get("height") or mask.shape[0]),
            "raw_box_xyxy": raw_box,
            "tight_box_xyxy": tight_box,
            "polygons": poly_res.polygons,
            "poly_meta": {
                "cc_policy": cc_policy,
                "nmax": int(nmax),
                "n_vertices": poly_res.n_vertices,
                "has_hole": False,
                "fidelity_iou": float(poly_res.fidelity_iou),
                "source": "rrsis-d_coco_rle",
            },
            "category": cat_name,
            "category_id": cat_id,
            "notes": None,
        }

        writers[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["total"] += 1
        stats[split] += 1
        if cat_name is not None:
            stats["categories"][cat_name] += 1
        stats["expr_len"].append(len(expr.split()))
        stats["bbox_area"].append(max(0, tight_box[2] - tight_box[0]) * max(0, tight_box[3] - tight_box[1]))
        stats["fidelity"].append(float(poly_res.fidelity_iou))

    for f in writers.values():
        f.close()
    bad_f.close()

    import numpy as np

    expr_lens = np.asarray(stats["expr_len"], dtype=np.int32) if stats["expr_len"] else np.asarray([], dtype=np.int32)
    bbox_areas = np.asarray(stats["bbox_area"], dtype=np.float64) if stats["bbox_area"] else np.asarray([], dtype=np.float64)
    fidelity = np.asarray(stats["fidelity"], dtype=np.float64) if stats["fidelity"] else np.asarray([], dtype=np.float64)

    meta_stats: dict[str, Any] = {
        "total_samples": int(stats["total"]),
        "train_samples": int(stats["train"]),
        "val_samples": int(stats["val"]),
        "test_samples": int(stats["test"]),
        "bad_samples": int(stats["bad"]),
        "empty_mask": int(stats["empty_mask"]),
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
            "p50": float(np.percentile(bbox_areas, 50)) if bbox_areas.size else 0.0,
            "p90": float(np.percentile(bbox_areas, 90)) if bbox_areas.size else 0.0,
            "min": float(bbox_areas.min()) if bbox_areas.size else 0.0,
            "max": float(bbox_areas.max()) if bbox_areas.size else 0.0,
        },
        "polygon_fidelity_iou": {
            "mean": float(fidelity.mean()) if fidelity.size else 0.0,
            "p10": float(np.percentile(fidelity, 10)) if fidelity.size else 0.0,
            "p50": float(np.percentile(fidelity, 50)) if fidelity.size else 0.0,
            "p90": float(np.percentile(fidelity, 90)) if fidelity.size else 0.0,
            "min": float(fidelity.min()) if fidelity.size else 0.0,
        },
        "has_mask": True,
        "polygon_source": "mask_to_polygons",
        "img_root": str(img_root),
        "instances": str(instances_path),
        "refs": str(refs_path),
        "nmax": int(nmax),
        "cc_policy": cc_policy,
        "approx_epsilon": approx_epsilon,
        "export_masks": bool(export_masks),
        "limit": limit,
    }

    with (out_dir / "meta_stats.json").open("w", encoding="utf-8") as f:
        json.dump(meta_stats, f, indent=2, ensure_ascii=False)

    return meta_stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert RRSIS-D to JSONL")
    ap.add_argument("--root", required=True, type=Path, help="Dataset root")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    ap.add_argument("--img-root", default=None, type=Path, help="Images directory")
    ap.add_argument("--nmax", default=80, type=int, help="Max polygon vertices")
    ap.add_argument("--cc-policy", default="largest", choices=["largest", "multi"], help="Connected components policy")
    ap.add_argument(
        "--approx-epsilon",
        default=None,
        type=float,
        help="cv2.approxPolyDP epsilon (None to disable)",
    )
    ap.add_argument("--export-masks", action="store_true", help="Write PNG masks under <out-dir>/masks")
    ap.add_argument("--limit", default=None, type=int, help="Convert only first N samples (debug)")
    args = ap.parse_args()

    meta = convert_rrsis_d(
        args.root,
        args.out_dir,
        img_root=args.img_root,
        nmax=args.nmax,
        cc_policy=args.cc_policy,
        approx_epsilon=args.approx_epsilon,
        export_masks=args.export_masks,
        limit=args.limit,
    )
    print("\n=== RRSIS-D conversion complete ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
