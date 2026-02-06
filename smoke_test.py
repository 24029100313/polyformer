"""
PolyFormer 环境测试脚本 - smoke_test.py
用于验证环境配置是否正确
"""

import sys
print("=" * 60)
print("PolyFormer 环境测试")
print("=" * 60)

# 1. 基础环境检查
print("\n[1/5] 基础环境检查")
print(f"  Python 版本: {sys.version}")

import torch
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. 关键依赖检查
print("\n[2/5] 关键依赖检查")
try:
    import fairseq
    print("  ✓ fairseq")
except ImportError as e:
    print(f"  ✗ fairseq: {e}")

try:
    import cv2
    print("  ✓ opencv-python")
except ImportError as e:
    print(f"  ✗ opencv-python: {e}")

try:
    import timm
    print("  ✓ timm")
except ImportError as e:
    print(f"  ✗ timm: {e}")

try:
    import einops
    print("  ✓ einops")
except ImportError as e:
    print(f"  ✗ einops: {e}")

try:
    from skimage import draw
    print("  ✓ scikit-image")
except ImportError as e:
    print(f"  ✗ scikit-image: {e}")

try:
    from PIL import Image
    print("  ✓ Pillow")
except ImportError as e:
    print(f"  ✗ Pillow: {e}")

# 3. 项目模块检查
print("\n[3/5] 项目模块检查")
try:
    from bert.tokenization_bert import BertTokenizer
    print("  ✓ bert.tokenization_bert")
except ImportError as e:
    print(f"  ✗ bert.tokenization_bert: {e}")

try:
    from models.polyformer import PolyFormerModel
    print("  ✓ models.polyformer")
except ImportError as e:
    print(f"  ✗ models.polyformer: {e}")

try:
    from tasks.refcoco import RefcocoTask
    print("  ✓ tasks.refcoco")
except ImportError as e:
    print(f"  ✗ tasks.refcoco: {e}")

try:
    from data.refcoco_dataset import RefcocoDataset
    print("  ✓ data.refcoco_dataset")
except ImportError as e:
    print(f"  ✗ data.refcoco_dataset: {e}")

try:
    from utils.checkpoint_utils import load_model_ensemble_and_task
    print("  ✓ utils.checkpoint_utils")
except ImportError as e:
    print(f"  ✗ utils.checkpoint_utils: {e}")

# 4. 权重文件检查
print("\n[4/5] 权重文件检查")
import os

weight_files = {
    "pretrained_weights/swin_base_patch4_window12_384_22k.pth": "Swin-base",
    "pretrained_weights/swin_large_patch4_window12_384_22k.pth": "Swin-large",
    "pretrained_weights/bert-base-uncased-pytorch_model.bin": "BERT-base",
    "weights/polyformer_b_pretrain.pt": "PolyFormer-B 预训练",
    "weights/polyformer_l_pretrain.pt": "PolyFormer-L 预训练",
    "weights/polyformer_l_refcocog.pt": "PolyFormer-L RefCOCOg",
}

for path, name in weight_files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {name}: {path} (未找到)")

# 5. 简单前向测试
print("\n[5/5] 简单前向测试")
try:
    # 测试 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_text = "a cat sitting on a table"
    tokens = tokenizer.encode(test_text)
    print(f"  ✓ BERT Tokenizer: '{test_text}' -> {len(tokens)} tokens")
    
    # 测试图像变换
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建一个测试图像
    import numpy as np
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    img_tensor = transform(test_img)
    print(f"  ✓ Image Transform: (480, 640, 3) -> {tuple(img_tensor.shape)}")
    
    # 测试 polygon 工具函数
    from data.poly_utils import string_to_polygons, polygons_to_string
    test_poly_str = "0.1,0.2,0.3,0.4,0.5,0.6"
    polygons = string_to_polygons(test_poly_str)
    print(f"  ✓ Polygon Utils: string_to_polygons works")
    
except Exception as e:
    print(f"  ✗ 前向测试失败: {e}")

print("\n" + "=" * 60)
print("测试完成！如果所有项目都显示 ✓，说明环境配置正确。")
print("=" * 60)

