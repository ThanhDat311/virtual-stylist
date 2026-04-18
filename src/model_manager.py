"""
Model Manager module for Virtual Stylist.
Quản lý vòng đời của model Siamese: versioning, cập nhật, và rollback.
Cho phép upload model mới từ Colab mà không cần restart app.
"""

import os
import json
import shutil
import hashlib
from datetime import datetime


# Thư mục chứa tất cả model versions
MODELS_DIR = "models"
MODEL_REGISTRY_FILE = os.path.join(MODELS_DIR, "registry.json")
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "siamese_best.pt")
BACKUP_DIR = os.path.join(MODELS_DIR, "backups")


def _compute_md5(filepath: str) -> str:
    """Tính MD5 hash của file để verify tính toàn vẹn."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry() -> dict:
    """Đọc file registry.json. Tạo mới nếu chưa có."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    if not os.path.exists(MODEL_REGISTRY_FILE):
        # Bootstrap registry từ model hiện có
        registry = {"active_version": None, "versions": []}

        if os.path.exists(ACTIVE_MODEL_PATH):
            ts = datetime.fromtimestamp(os.path.getmtime(ACTIVE_MODEL_PATH))
            file_size = os.path.getsize(ACTIVE_MODEL_PATH)
            md5 = _compute_md5(ACTIVE_MODEL_PATH)
            entry = {
                "version": "v1.0.0",
                "filename": "siamese_best.pt",
                "created_at": ts.isoformat(),
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "md5": md5,
                "description": "Model gốc (bootstrap từ file hiện có)",
                "training_info": {},
            }
            registry["versions"] = [entry]
            registry["active_version"] = "v1.0.0"

        _save_registry(registry)
        return registry

    with open(MODEL_REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict):
    """Ghi registry.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
class ModelManager:
    """
    Quản lý versions của model Siamese.

    Cách dùng:
        mgr = ModelManager()
        info = mgr.get_current_model_info()
        success, msg = mgr.update_model("/path/to/new_model.pt", description="Epoch 30 - acc 0.87")
        mgr.rollback("v1.0.0")
    """

    def __init__(self):
        self.registry = _load_registry()

    # ── Truy vấn ──────────────────────────────────────────────────────────────
    def get_current_model_info(self) -> dict:
        """Trả về thông tin của model đang được dùng."""
        active = self.registry.get("active_version")
        if not active:
            return {"status": "no_model", "message": "Chưa có model nào được đăng ký."}

        for v in self.registry["versions"]:
            if v["version"] == active:
                v["status"] = "active"
                return v

        return {"status": "error", "message": "Active version không tìm thấy trong registry."}

    def list_versions(self) -> list:
        """Trả về danh sách tất cả versions đã từng dùng."""
        return self.registry.get("versions", [])

    def get_active_model_path(self) -> str:
        """Trả về đường dẫn tuyệt đối của model đang active."""
        return os.path.abspath(ACTIVE_MODEL_PATH)

    # ── Cập nhật model ────────────────────────────────────────────────────────
    def update_model(self, new_model_path: str, description: str = "", training_info: dict = None) -> tuple[bool, str]:
        """
        Cập nhật model mới.

        Args:
            new_model_path: Đường dẫn đến file .pt mới (từ Colab export, v.v.)
            description: Mô tả ngắn về model này (vd: "Epoch 50 - val_loss 0.23")
            training_info: Dict chứa thông tin training (accuracy, loss, epoch, v.v.)

        Returns:
            (success: bool, message: str)
        """
        try:
            if not os.path.exists(new_model_path):
                return False, f"❌ File không tồn tại: {new_model_path}"

            if not new_model_path.endswith(".pt"):
                return False, "❌ File phải có đuôi .pt (PyTorch checkpoint)"

            file_size = os.path.getsize(new_model_path)
            if file_size < 1024:  # < 1KB → likely corrupt
                return False, "❌ File quá nhỏ, có thể bị hỏng."

            # 1. Backup model hiện tại
            if os.path.exists(ACTIVE_MODEL_PATH):
                current_info = self.get_current_model_info()
                backup_name = f"siamese_{current_info.get('version', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                backup_path = os.path.join(BACKUP_DIR, backup_name)
                shutil.copy2(ACTIVE_MODEL_PATH, backup_path)

            # 2. Copy model mới vào vị trí active
            shutil.copy2(new_model_path, ACTIVE_MODEL_PATH)

            # 3. Xác định version mới
            versions = self.registry.get("versions", [])
            new_version_num = len(versions) + 1
            new_version = f"v{new_version_num}.0.0"

            # 4. Tạo entry mới trong registry
            new_entry = {
                "version": new_version,
                "filename": "siamese_best.pt",
                "created_at": datetime.now().isoformat(),
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "md5": _compute_md5(ACTIVE_MODEL_PATH),
                "description": description or f"Model cập nhật lúc {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                "training_info": training_info or {},
            }

            self.registry["versions"].append(new_entry)
            self.registry["active_version"] = new_version
            _save_registry(self.registry)

            return True, (
                f"✅ Model đã được cập nhật thành công!\n"
                f"   Version: {new_version}\n"
                f"   Kích thước: {new_entry['file_size_mb']} MB\n"
                f"   MD5: {new_entry['md5'][:8]}...\n\n"
                f"⚠️  Nhấn nút 'Reload Model' để áp dụng model mới vào engine."
            )

        except Exception as e:
            return False, f"❌ Lỗi khi cập nhật model: {str(e)}"

    # ── Rollback ──────────────────────────────────────────────────────────────
    def rollback(self, target_version: str) -> tuple[bool, str]:
        """
        Rollback về một version cũ hơn từ thư mục backup.

        Args:
            target_version: Version cần khôi phục (vd: "v1.0.0")

        Returns:
            (success: bool, message: str)
        """
        try:
            versions = self.registry.get("versions", [])
            target_entry = next((v for v in versions if v["version"] == target_version), None)
            if not target_entry:
                return False, f"❌ Không tìm thấy version '{target_version}' trong registry."

            # Tìm file backup
            backup_files = os.listdir(BACKUP_DIR) if os.path.exists(BACKUP_DIR) else []
            matching = [f for f in backup_files if target_version.replace(".", "_") in f or target_version in f]

            if not matching and target_version == versions[0]["version"]:
                # Không có backup cho v1 → model hiện tại là v1 (chưa từng update)
                return False, f"❌ Không tìm thấy file backup cho '{target_version}'."

            if not matching:
                return False, f"❌ Không tìm thấy file backup cho '{target_version}'."

            backup_path = os.path.join(BACKUP_DIR, sorted(matching)[-1])
            shutil.copy2(backup_path, ACTIVE_MODEL_PATH)
            self.registry["active_version"] = target_version
            _save_registry(self.registry)

            return True, (
                f"✅ Đã rollback về {target_version} thành công.\n"
                f"⚠️  Nhấn nút 'Reload Model' để áp dụng."
            )

        except Exception as e:
            return False, f"❌ Lỗi khi rollback: {str(e)}"

    def delete_old_backups(self, keep_last: int = 3) -> str:
        """Xóa bớt backup cũ, chỉ giữ lại `keep_last` bản gần nhất."""
        if not os.path.exists(BACKUP_DIR):
            return "Không có backup nào."
        files = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join(BACKUP_DIR, x))
        )
        to_delete = files[:-keep_last] if len(files) > keep_last else []
        for f in to_delete:
            os.remove(os.path.join(BACKUP_DIR, f))
        return f"🗑️ Đã xóa {len(to_delete)} backup cũ, giữ lại {min(keep_last, len(files))} bản."
