#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sniff RRSIS-D dataset structure and format."""

import json
import pickle
import sys
from pathlib import Path

def main():
    root = Path("datasets/rrsis_d/rrsisd")
    
    # Load instances.json
    with open(root / "instances.json", "r") as f:
        data = json.load(f)
    
    # Load refs
    with open(root / "refs(unc).p", "rb") as f:
        refs = pickle.load(f)
    
    print("=" * 60)
    print("RRSIS-D Dataset Sniff Report")
    print("=" * 60)
    
    # Categories
    print("\n=== Categories ===")
    for c in data["categories"]:
        print(f"  id={c['id']}: {c['name']}")
    print(f"  Total: {len(data['categories'])} categories")
    
    # Images
    print(f"\n=== Images ===")
    print(f"  Total: {len(data['images'])}")
    img0 = data["images"][0]
    print(f"  First: {img0}")
    sizes = set()
    for im in data["images"]:
        sizes.add((im["width"], im["height"]))
    print(f"  Unique sizes: {sizes}")
    
    # Annotations
    print(f"\n=== Annotations ===")
    print(f"  Total: {len(data['annotations'])}")
    ann0 = data["annotations"][0]
    print(f"  Keys: {list(ann0.keys())}")
    print(f"  bbox: {ann0['bbox']}")
    print(f"  area: {ann0['area']}")
    
    # Check bbox format: [x1,y1,x2,y2] or [x,y,w,h]?
    print(f"\n=== Bbox format check ===")
    for a in data["annotations"][:5]:
        b = a["bbox"]
        img = next(im for im in data["images"] if im["id"] == a["image_id"])
        print(f"  img_id={a['image_id']}, bbox={b}, img={img['width']}x{img['height']}")
        # If [x1,y1,x2,y2]: x2>x1 and y2>y1
        # If [x,y,w,h]: b[2] and b[3] are sizes
        if b[2] > b[0] and b[3] > b[1]:
            fmt = "likely [x1,y1,x2,y2]"
        else:
            fmt = "likely [x,y,w,h]"
        print(f"    => {fmt}")
    
    # Segmentation format
    print(f"\n=== Segmentation ===")
    seg = ann0["segmentation"]
    print(f"  Type: {type(seg)}")
    if isinstance(seg, list) and len(seg) > 0:
        seg0 = seg[0]
        print(f"  Element type: {type(seg0)}")
        if isinstance(seg0, dict):
            print(f"  Keys: {list(seg0.keys())}")
            print(f"  RLE size: {seg0.get('size')}")
            counts = seg0.get("counts", "")
            print(f"  Counts type: {type(counts)}, length: {len(counts) if isinstance(counts, str) else 'N/A'}")
    
    # Check if all annotations have segmentation
    has_seg = sum(1 for a in data["annotations"] if a.get("segmentation"))
    empty_seg = sum(1 for a in data["annotations"] if not a.get("segmentation"))
    print(f"  Has segmentation: {has_seg}")
    print(f"  Empty segmentation: {empty_seg}")
    
    # Refs
    print(f"\n=== Refs ===")
    print(f"  Total: {len(refs)}")
    print(f"  Keys: {list(refs[0].keys())}")
    
    # Split counts
    from collections import Counter
    splits = Counter(r["split"] for r in refs)
    print(f"  Splits: {dict(splits)}")
    
    # Sample expressions
    print(f"\n=== Sample expressions ===")
    for i in [0, 100, 500, 1000, 5000]:
        r = refs[i]
        sents = r["sentences"]
        if isinstance(sents, str):
            sents = eval(sents)
        for s in sents:
            print(f"  [{r['split']}] img={r['file_name']}: {s['sent']}")
    
    # Check if ann_id matches annotation id
    print(f"\n=== ID mapping check ===")
    ann_ids = {a["id"] for a in data["annotations"]}
    ref_ann_ids = set()
    for r in refs:
        aid = r["ann_id"]
        if isinstance(aid, str):
            aid = int(aid)
        ref_ann_ids.add(aid)
    
    print(f"  Annotation IDs: {len(ann_ids)}")
    print(f"  Ref ann_ids: {len(ref_ann_ids)}")
    overlap = ann_ids & ref_ann_ids
    print(f"  Overlap: {len(overlap)}")
    missing_in_ann = ref_ann_ids - ann_ids
    missing_in_ref = ann_ids - ref_ann_ids
    print(f"  In refs but not in annotations: {len(missing_in_ann)}")
    print(f"  In annotations but not in refs: {len(missing_in_ref)}")


if __name__ == "__main__":
    main()

