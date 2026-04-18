"""
save_dataset_to_drive.py
========================
Script giải nén dataset H&M Personalized Fashion Recommendations
và lưu trữ lên Google Drive (dùng trong Google Colab).

Cấu trúc thư mục sau khi chạy:
    /content/drive/MyDrive/HM_Dataset/
    ├── articles.csv
    ├── customers.csv
    ├── transactions_train.csv
    ├── sample_submission.csv
    └── images/
        └── ...
"""

import os
import zipfile
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Mount Google Drive
# ─────────────────────────────────────────────
from google.colab import drive

print("🔗 Đang mount Google Drive...")
drive.mount("/content/drive", force_remount=False)
print("✅ Google Drive đã được mount thành công!\n")


# ─────────────────────────────────────────────
# 2. Cấu hình đường dẫn
# ─────────────────────────────────────────────
# Đường dẫn file zip gốc (trong Colab hoặc Kaggle)
ZIP_PATH = "/content/sample_data/h-and-m-personalized-fashion-recommendations.zip"

# Thư mục tạm để giải nén
EXTRACT_TEMP_DIR = "/content/hm_extracted"

# Thư mục đích trên Google Drive
DRIVE_DEST_DIR = "/content/drive/MyDrive/HM_Dataset"


# ─────────────────────────────────────────────
# 3. Kiểm tra file zip tồn tại
# ─────────────────────────────────────────────
if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"❌ Không tìm thấy file zip tại: {ZIP_PATH}\n"
        "   Vui lòng kiểm tra lại đường dẫn ZIP_PATH."
    )

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 ** 2)
print(f"📦 Tìm thấy file zip: {ZIP_PATH}")
print(f"   Kích thước: {zip_size_mb:.1f} MB\n")


# ─────────────────────────────────────────────
# 4. Giải nén dataset
# ─────────────────────────────────────────────
print(f"📂 Đang giải nén vào: {EXTRACT_TEMP_DIR}")
os.makedirs(EXTRACT_TEMP_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    members = zf.namelist()
    total = len(members)
    print(f"   Tổng số file trong zip: {total}")
    
    for i, member in enumerate(members, 1):
        zf.extract(member, EXTRACT_TEMP_DIR)
        # Hiển thị tiến trình mỗi 500 file
        if i % 500 == 0 or i == total:
            print(f"   [{i}/{total}] {member}")

print("✅ Giải nén hoàn tất!\n")


# ─────────────────────────────────────────────
# 5. Sao chép lên Google Drive
# ─────────────────────────────────────────────
print(f"☁️  Đang sao chép lên Google Drive...")
print(f"   Đích: {DRIVE_DEST_DIR}\n")

os.makedirs(DRIVE_DEST_DIR, exist_ok=True)

extracted_path = Path(EXTRACT_TEMP_DIR)
dest_path = Path(DRIVE_DEST_DIR)

# Liệt kê tất cả file/thư mục cần copy
all_items = list(extracted_path.rglob("*"))
total_items = len(all_items)
copied = 0
skipped = 0

for item in all_items:
    relative = item.relative_to(extracted_path)
    target = dest_path / relative
    
    if item.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        # Bỏ qua nếu file đã tồn tại và cùng kích thước (tránh upload lại)
        if target.exists() and target.stat().st_size == item.stat().st_size:
            skipped += 1
            continue
        
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
        
        if copied % 500 == 0:
            print(f"   Đã sao chép: {copied} file...")

print(f"\n✅ Hoàn tất sao chép lên Google Drive!")
print(f"   📁 Đã copy : {copied} file")
print(f"   ⏭️  Đã bỏ qua (đã tồn tại): {skipped} file")


# ─────────────────────────────────────────────
# 6. Xác minh kết quả trên Drive
# ─────────────────────────────────────────────
print(f"\n📋 Kiểm tra nội dung thư mục trên Drive:")
print(f"   {DRIVE_DEST_DIR}")
print("   " + "─" * 50)

total_size = 0
for item in sorted(dest_path.iterdir()):
    if item.is_file():
        size_mb = item.stat().st_size / (1024 ** 2)
        total_size += size_mb
        print(f"   📄 {item.name:<40} {size_mb:>8.2f} MB")
    elif item.is_dir():
        n_files = sum(1 for _ in item.rglob("*") if _.is_file())
        print(f"   📁 {item.name:<40} ({n_files} files)")

print("   " + "─" * 50)
print(f"   Tổng dung lượng file CSV: {total_size:.1f} MB")


# ─────────────────────────────────────────────
# 7. Dọn dẹp thư mục tạm (tuỳ chọn)
# ─────────────────────────────────────────────
CLEANUP_TEMP = True   # Đặt False nếu muốn giữ lại

if CLEANUP_TEMP:
    shutil.rmtree(EXTRACT_TEMP_DIR, ignore_errors=True)
    print(f"\n🧹 Đã xóa thư mục tạm: {EXTRACT_TEMP_DIR}")
else:
    print(f"\n💾 Thư mục tạm được giữ lại tại: {EXTRACT_TEMP_DIR}")

print("\n🎉 Script hoàn thành! Dataset đã sẵn sàng trên Google Drive.")
print(f"   Đường dẫn Drive: {DRIVE_DEST_DIR}")
