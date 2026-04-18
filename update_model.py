#!/usr/bin/env python
"""
update_model.py — Script CLI để cập nhật model Siamese mới.

Sử dụng:
    python update_model.py <path_to_new_model.pt> [--desc "Mô tả"] [--epoch 50] [--loss 0.21] [--acc 0.89]

Ví dụ:
    python update_model.py D:/Downloads/siamese_best.pt --desc "Epoch 50 colab run" --epoch 50 --loss 0.21 --acc 0.89
    python update_model.py D:/Downloads/siamese_best.pt
"""

import argparse
import sys
import os

# Đảm bảo có thể import từ thư mục gốc project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description="Cập nhật model Siamese mới vào Virtual Stylist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_path",
        help="Đường dẫn đến file model mới (.pt)",
    )
    parser.add_argument(
        "--desc",
        default="",
        help="Mô tả về model này (vd: 'Epoch 50 - val_loss 0.21')",
    )
    parser.add_argument("--epoch", default="", help="Số epoch đã train")
    parser.add_argument("--loss", default="", help="Validation loss")
    parser.add_argument("--acc", default="", help="Accuracy trên tập test")

    args = parser.parse_args()

    print("=" * 60)
    print("  Virtual Stylist — Model Updater")
    print("=" * 60)
    print(f"  File mới  : {args.model_path}")
    print(f"  Mô tả     : {args.desc or '(không có)'}")
    print()

    training_info = {}
    if args.epoch:
        training_info["epoch"] = args.epoch
    if args.loss:
        training_info["val_loss"] = args.loss
    if args.acc:
        training_info["accuracy"] = args.acc

    mgr = ModelManager()

    # Hiển thị version hiện tại
    current = mgr.get_current_model_info()
    print(f"  Version hiện tại: {current.get('version', 'N/A')} — {current.get('description', '')}")
    print()

    # Thực hiện update
    success, msg = mgr.update_model(
        args.model_path,
        description=args.desc,
        training_info=training_info,
    )

    print(msg)
    print()

    if success:
        print("✅ Xong! Để áp dụng vào app đang chạy:")
        print("   → Vào tab '⚙️ Model Manager' trong Gradio UI")
        print("   → Nhấn nút '🔄 Reload Model vào Engine'")
        print()
        print("   HOẶC restart app: python app.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
