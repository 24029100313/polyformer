#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_refdior.py - 将 DIOR-RSVG 数据集转换为 PolyFormer 训练用的 JSONL 格式

用法:
    python tools/refdior/convert_refdior.py \
        --root datasets/DIOR-RSVG \
        --out-dir datasets/refdior/processed

输出:
    - refdior_train.jsonl
    - refdior_val.jsonl  
    - refdior_test.jsonl
    - meta_stats.json
    - bad_samples.jsonl
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


def parse_xml(xml_path: str) -> Dict:
    """解析单个 XML annotation 文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        desc_elem = obj.find('description')
        description = desc_elem.text if desc_elem is not None else ""
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'description': description
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def bbox_to_polygon(bbox: List[int]) -> List[List[int]]:
    """将 bbox [xmin, ymin, xmax, ymax] 转换为顺时针矩形 polygon
    
    顺序：左上 -> 右上 -> 右下 -> 左下（顺时针，起点为左上角，最接近原点）
    """
    xmin, ymin, xmax, ymax = bbox
    return [
        [xmin, ymin],  # 左上 (最接近原点)
        [xmax, ymin],  # 右上
        [xmax, ymax],  # 右下
        [xmin, ymax],  # 左下
    ]


def build_sample_index(ann_dir: str) -> List[Tuple[str, int]]:
    """构建全局样本索引: [(xml_file, object_idx), ...]
    
    按文件名排序，然后展开每个文件的所有 objects
    """
    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
    
    sample_index = []
    for xml_file in xml_files:
        xml_path = os.path.join(ann_dir, xml_file)
        data = parse_xml(xml_path)
        for obj_idx in range(len(data['objects'])):
            sample_index.append((xml_file, obj_idx))
    
    return sample_index


def load_split_ids(split_file: str) -> set:
    """加载 split 文件中的 ID"""
    with open(split_file, 'r') as f:
        return set(int(line.strip()) for line in f if line.strip())


def convert_dataset(
    root_dir: str,
    out_dir: str,
    img_root: Optional[str] = None
) -> Dict:
    """转换整个数据集"""
    
    ann_dir = os.path.join(root_dir, 'Annotations')
    if img_root is None:
        img_root = os.path.join(root_dir, 'JPEGImages')
    
    # 加载 split
    train_ids = load_split_ids(os.path.join(root_dir, 'train.txt'))
    val_ids = load_split_ids(os.path.join(root_dir, 'val.txt'))
    test_ids = load_split_ids(os.path.join(root_dir, 'test.txt'))
    
    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # 构建样本索引
    print("Building sample index...")
    sample_index = build_sample_index(ann_dir)
    print(f"Total samples: {len(sample_index)}")
    
    # 验证
    all_ids = train_ids | val_ids | test_ids
    assert len(sample_index) == len(all_ids), \
        f"Sample count mismatch: {len(sample_index)} vs {len(all_ids)}"
    
    # 准备输出
    os.makedirs(out_dir, exist_ok=True)
    
    splits = {
        'train': (train_ids, open(os.path.join(out_dir, 'refdior_train.jsonl'), 'w', encoding='utf-8')),
        'val': (val_ids, open(os.path.join(out_dir, 'refdior_val.jsonl'), 'w', encoding='utf-8')),
        'test': (test_ids, open(os.path.join(out_dir, 'refdior_test.jsonl'), 'w', encoding='utf-8')),
    }
    
    bad_samples_f = open(os.path.join(out_dir, 'bad_samples.jsonl'), 'w', encoding='utf-8')
    
    # 统计
    stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0,
        'bad': 0,
        'categories': defaultdict(int),
        'expr_lengths': [],
        'bbox_sizes': [],
    }
    
    # 缓存解析结果
    xml_cache = {}
    
    # 转换
    print("Converting samples...")
    for global_idx, (xml_file, obj_idx) in enumerate(tqdm(sample_index)):
        # 解析 XML（带缓存）
        if xml_file not in xml_cache:
            xml_path = os.path.join(ann_dir, xml_file)
            xml_cache[xml_file] = parse_xml(xml_path)
        
        data = xml_cache[xml_file]
        obj = data['objects'][obj_idx]
        
        # 构造图像路径
        image_path = os.path.join(img_root, data['filename'])
        
        # 检查 expression
        expr = obj['description']
        if not expr or not expr.strip():
            bad_sample = {
                'sample_id': str(global_idx),
                'reason': 'empty_expression',
                'xml_file': xml_file,
                'obj_idx': obj_idx
            }
            bad_samples_f.write(json.dumps(bad_sample, ensure_ascii=False) + '\n')
            stats['bad'] += 1
            continue
        
        # bbox 转 polygon
        bbox = obj['bbox']
        polygon = bbox_to_polygon(bbox)
        
        # 计算 tight_box (对于矩形 polygon，tight_box = raw_box)
        tight_box = bbox.copy()
        
        # 构造 JSONL 样本
        sample = {
            'sample_id': str(global_idx),
            'split': None,  # 稍后填充
            'image_path': image_path,
            'mask_path': None,  # DIOR-RSVG 没有 mask
            'image_id': xml_file.replace('.xml', ''),
            
            'expr': expr.strip(),
            'img_w': data['width'],
            'img_h': data['height'],
            
            'raw_box_xyxy': bbox,
            'tight_box_xyxy': tight_box,
            
            'polygons': [polygon],  # 单个矩形 polygon
            
            'poly_meta': {
                'cc_policy': 'bbox_as_polygon',
                'nmax': 4,
                'n_vertices': [4],
                'has_hole': False,
                'fidelity_iou': 1.0,  # bbox 完美匹配自身
                'source': 'dior-rsvg'
            },
            
            'category': obj['name'],
            'notes': None
        }
        
        # 确定 split
        if global_idx in train_ids:
            sample['split'] = 'train'
            splits['train'][1].write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['train'] += 1
        elif global_idx in val_ids:
            sample['split'] = 'val'
            splits['val'][1].write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['val'] += 1
        elif global_idx in test_ids:
            sample['split'] = 'test'
            splits['test'][1].write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['test'] += 1
        else:
            bad_sample = {
                'sample_id': str(global_idx),
                'reason': 'not_in_any_split',
                'xml_file': xml_file,
                'obj_idx': obj_idx
            }
            bad_samples_f.write(json.dumps(bad_sample, ensure_ascii=False) + '\n')
            stats['bad'] += 1
            continue
        
        stats['total'] += 1
        stats['categories'][obj['name']] += 1
        stats['expr_lengths'].append(len(expr.split()))
        
        # bbox 面积
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        stats['bbox_sizes'].append(w * h)
    
    # 关闭文件
    for split_name, (_, f) in splits.items():
        f.close()
    bad_samples_f.close()
    
    # 计算统计信息
    import numpy as np
    expr_lens = np.array(stats['expr_lengths'])
    bbox_sizes = np.array(stats['bbox_sizes'])
    
    meta_stats = {
        'total_samples': stats['total'],
        'train_samples': stats['train'],
        'val_samples': stats['val'],
        'test_samples': stats['test'],
        'bad_samples': stats['bad'],
        'num_images': len(xml_cache),
        'num_categories': len(stats['categories']),
        'categories': dict(stats['categories']),
        'expr_length_stats': {
            'mean': float(np.mean(expr_lens)),
            'std': float(np.std(expr_lens)),
            'min': int(np.min(expr_lens)),
            'max': int(np.max(expr_lens)),
            'p50': float(np.percentile(expr_lens, 50)),
            'p90': float(np.percentile(expr_lens, 90)),
            'p99': float(np.percentile(expr_lens, 99)),
        },
        'bbox_size_stats': {
            'mean': float(np.mean(bbox_sizes)),
            'std': float(np.std(bbox_sizes)),
            'min': int(np.min(bbox_sizes)),
            'max': int(np.max(bbox_sizes)),
        },
        'image_size': {'width': 800, 'height': 800},  # DIOR 固定尺寸
        'has_mask': False,
        'polygon_source': 'bbox_as_rectangle',
    }
    
    # 保存 meta_stats
    with open(os.path.join(out_dir, 'meta_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_stats, f, indent=2, ensure_ascii=False)
    
    return meta_stats


def main():
    parser = argparse.ArgumentParser(description='Convert DIOR-RSVG to JSONL format')
    parser.add_argument('--root', type=str, required=True,
                        help='Path to DIOR-RSVG root directory')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for JSONL files')
    parser.add_argument('--img-root', type=str, default=None,
                        help='Path to JPEGImages directory (default: <root>/JPEGImages)')
    
    args = parser.parse_args()
    
    print(f"Converting DIOR-RSVG from: {args.root}")
    print(f"Output to: {args.out_dir}")
    
    meta_stats = convert_dataset(args.root, args.out_dir, args.img_root)
    
    print("\n=== Conversion Complete ===")
    print(f"Total: {meta_stats['total_samples']}")
    print(f"Train: {meta_stats['train_samples']}")
    print(f"Val: {meta_stats['val_samples']}")
    print(f"Test: {meta_stats['test_samples']}")
    print(f"Bad: {meta_stats['bad_samples']}")
    print(f"Categories: {meta_stats['num_categories']}")
    print(f"Expression length (mean): {meta_stats['expr_length_stats']['mean']:.1f} words")


if __name__ == '__main__':
    main()

