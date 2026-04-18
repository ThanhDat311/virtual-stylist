import os
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
from src.closet_manager import ClosetManager, CLOSET_DIR, CSV_PATH
from src.prompt_processor import process_prompt
from src.siamese_engine import SiameseEngine
from src.model_manager import ModelManager

# ── Khởi tạo core modules ──────────────────────────────────────────────────────
closet_mgr = ClosetManager()
stylist_ai = SiameseEngine(model_path="models/siamese_best.pt")
model_mgr = ModelManager()


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_closet():
    if not os.path.exists(CSV_PATH):
        return []
    try:
        df = pd.read_csv(CSV_PATH)
        return [
            (
                os.path.join(CLOSET_DIR, row.filename),
                f"{row.role} | {row.color} | {row.material}",
            )
            for _, row in df.iterrows()
        ]
    except Exception:
        return []


# ── Tab 1: Tủ quần áo ─────────────────────────────────────────────────────────
def clear_closet_action():
    msg = closet_mgr.clear_closet()
    return [], msg


def upload_and_refresh(images):
    if images is None:
        return load_closet(), "Please select images."

    results = []
    for img in images:
        metadata = closet_mgr.process_new_item(img)
        results.append(f"Uploaded: {metadata['role']} ({metadata['color']})")

    return load_closet(), "\n".join(results)


# ── Tab 2: AI Stylist ──────────────────────────────────────────────────────────
def get_styling_recommendation(prompt_text):
    if not os.path.exists(CSV_PATH):
        return None, "Closet is empty. Please upload some clothes first!"

    df = pd.read_csv(CSV_PATH)
    if len(df) == 0:
        return None, "Closet is empty!"

    # 1. Xử lý prompt
    intent = process_prompt(prompt_text)

    # 2. Lọc theo intent
    filtered_df = df.copy()
    if intent["role"]:
        filtered_df = df[df["role"] == intent["role"]]
    if intent["color"]:
        filtered_df = filtered_df[filtered_df["color"] == intent["color"]]
    if intent["material"]:
        filtered_df = filtered_df[filtered_df["material"] == intent["material"]]

    if len(filtered_df) == 0:
        return None, f"Không tìm thấy món đồ nào phù hợp với: '{prompt_text}'"

    # 3. Chọn starter item
    starter_item = filtered_df.iloc[0]
    starter_emb = np.load(starter_item.embedding_path)
    starter_img_path = os.path.join(CLOSET_DIR, starter_item.filename)

    # 4. Dùng SiameseEngine (kết hợp Siamese + Color Harmony)
    target_roles = ["UPPER", "LOWER", "SHOES", "ACCESSORIES"]
    if starter_item.role in target_roles:
        target_roles.remove(starter_item.role)

    outfit_items = [
        (
            starter_img_path,
            f"STARTER: {starter_item.role} ({starter_item.color})",
        )
    ]

    explanation_lines = [f"🎯 **Starter**: {starter_item.color} {starter_item.role}"]

    for role in target_roles:
        role_candidates = df[df["role"] == role]
        if len(role_candidates) == 0:
            continue

        candidate_embs = np.stack(
            [np.load(path) for path in role_candidates.embedding_path]
        )
        candidate_colors = role_candidates["color"].tolist()

        # ★ Ensemble scoring (Siamese + Color Harmony)
        scores = stylist_ai.rank_compatibility(
            query_emb=starter_emb,
            candidate_embs=candidate_embs,
            query_color=str(starter_item.color),
            candidate_colors=candidate_colors,
            color_weight=0.25,
        )

        best_idx = int(np.argmax(scores))
        best_item = role_candidates.iloc[best_idx]
        outfit_items.append(
            (
                os.path.join(CLOSET_DIR, best_item.filename),
                f"MATCH: {best_item.role} ({best_item.color}) — Score: {scores[best_idx]:.2f}",
            )
        )
        explanation_lines.append(
            f"👉 **{best_item.role}**: {best_item.color} (score: {scores[best_idx]:.2f})"
        )

    explanation = "\n".join(explanation_lines)
    return outfit_items, explanation


# ── Tab 3: Model Manager ───────────────────────────────────────────────────────
def get_model_info_display():
    """Trả về string hiển thị thông tin model hiện tại."""
    info = model_mgr.get_current_model_info()
    status = stylist_ai.get_model_status()

    lines = [
        "## 🤖 Thông tin Model Hiện Tại",
        "",
        f"**Version:** `{info.get('version', 'N/A')}`",
        f"**Mô tả:** {info.get('description', 'N/A')}",
        f"**Ngày tạo:** {info.get('created_at', 'N/A')[:19].replace('T', ' ')}",
        f"**Kích thước file:** {info.get('file_size_mb', 0)} MB",
        f"**MD5:** `{info.get('md5', 'N/A')[:12]}...`",
        "",
        "---",
        f"**Engine status:** {'✅ Model đã load' if status['loaded'] else '⚠️ Đang dùng fallback (cosine similarity + color harmony)'}",
        f"**Device:** `{status['device'].upper()}`",
    ]

    ti = info.get("training_info", {})
    if ti:
        lines += [
            "",
            "**Training Info:**",
            *[f"  - {k}: {v}" for k, v in ti.items()],
        ]

    return "\n".join(lines)


def get_versions_table():
    """Trả về DataFrame của các versions để hiển thị."""
    versions = model_mgr.list_versions()
    if not versions:
        return pd.DataFrame(columns=["Version", "Ngày tạo", "Size (MB)", "Mô tả"])

    active = model_mgr.registry.get("active_version", "")
    rows = []
    for v in versions:
        rows.append(
            {
                "Version": ("⭐ " if v["version"] == active else "") + v["version"],
                "Ngày tạo": v.get("created_at", "")[:19].replace("T", " "),
                "Size (MB)": v.get("file_size_mb", 0),
                "Mô tả": v.get("description", ""),
            }
        )
    return pd.DataFrame(rows)


def handle_model_upload(file_obj, description_text, epoch_text, loss_text, acc_text):
    """Xử lý khi user upload model mới qua Gradio."""
    if file_obj is None:
        return "⚠️ Chưa chọn file.", get_versions_table()

    training_info = {}
    if epoch_text.strip():
        training_info["epoch"] = epoch_text.strip()
    if loss_text.strip():
        training_info["val_loss"] = loss_text.strip()
    if acc_text.strip():
        training_info["accuracy"] = acc_text.strip()

    # file_obj từ Gradio (gr.File) là một đối tượng có thuộc tính .name
    src_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    success, msg = model_mgr.update_model(src_path, description=description_text, training_info=training_info)
    return msg, get_versions_table()


def handle_reload_model():
    """Hot-reload model đang active vào engine mà không restart app."""
    success, msg = stylist_ai.reload_model()
    # Refresh registry từ disk (để hiển thị version mới nhất)
    from src.model_manager import _load_registry
    model_mgr.registry = _load_registry()
    return msg, get_model_info_display()


def handle_rollback(version_str):
    """Rollback về version cũ."""
    version_str = version_str.strip().replace("⭐ ", "")
    if not version_str:
        return "⚠️ Chưa nhập version.", get_versions_table()
    success, msg = model_mgr.rollback(version_str)
    return msg, get_versions_table()


def handle_cleanup():
    msg = model_mgr.delete_old_backups(keep_last=3)
    return msg


# ── UI Construction ────────────────────────────────────────────────────────────
custom_css = """
.model-info-box { background: #1e293b; border-radius: 12px; padding: 16px; }
.version-badge { font-family: monospace; font-weight: bold; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="👗 Virtual Stylist AI",
    css=custom_css,
) as demo:
    gr.Markdown(
        """
        # 👗 My Personal Virtual Closet
        *Nâng cấp phong cách với AI phối đồ từ tủ quần áo của bạn.*
        """
    )

    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tabs():

        # ─── Tab 1: Tủ quần áo ──────────────────────────────────────────────
        with gr.Tab("🏠 Tủ Quần Áo"):
            with gr.Row():
                with gr.Column(scale=1):
                    uploader = gr.File(
                        label="Upload ảnh quần áo",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    upload_btn = gr.Button(
                        "📥 Xử lý & Thêm vào Tủ", variant="primary"
                    )
                    clear_btn = gr.Button("🗑️ Xóa toàn bộ Tủ", variant="stop")
                    status_log = gr.Textbox(label="Trạng thái", interactive=False)

                with gr.Column(scale=2):
                    closet_gallery = gr.Gallery(
                        label="Tủ Quần Áo Của Tôi",
                        value=load_closet(),
                        columns=4,
                        height="600px",
                    )

            upload_btn.click(
                upload_and_refresh,
                inputs=[uploader],
                outputs=[closet_gallery, status_log],
            )
            clear_btn.click(
                clear_closet_action, outputs=[closet_gallery, status_log]
            )

        # ─── Tab 2: AI Stylist ───────────────────────────────────────────────
        with gr.Tab("✨ AI Stylist"):
            gr.Markdown("### 💬 Hỏi AI stylist của bạn về outfit!")
            with gr.Row():
                prompt_input = gr.Textbox(
                    placeholder="VD: 'Phối cho tôi áo thun đen' hoặc 'I want an outfit with blue jeans'...",
                    label="Nhập yêu cầu của bạn",
                    lines=2,
                )
                style_btn = gr.Button("✨ Gợi ý Outfit", variant="primary", scale=0)

            output_msg = gr.Markdown("*Nhập yêu cầu rồi nhấn nút Gợi ý Outfit!*")
            outfit_gallery = gr.Gallery(
                label="Outfit Được Gợi Ý", columns=4, height="400px"
            )

            style_btn.click(
                get_styling_recommendation,
                inputs=[prompt_input],
                outputs=[outfit_gallery, output_msg],
            )

        # ─── Tab 3: ⚙️ Model Manager ────────────────────────────────────────
        with gr.Tab("⚙️ Model Manager"):
            gr.Markdown(
                """
                ### 🔄 Quản lý & Cập nhật Model AI

                **Cách update model từ Colab:**
                1. Xuất model từ Colab: `torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, 'siamese_best.pt')`
                2. Download file `.pt` về máy local
                3. Upload file đó ở đây → nhấn **"Cập nhật Model"**
                4. Nhấn **"🔄 Reload Model vào Engine"** để áp dụng ngay
                """
            )

            with gr.Row():
                # ── Cột trái: thông tin model ──────────────────────────────
                with gr.Column(scale=1):
                    model_info_md = gr.Markdown(
                        value=get_model_info_display(), label="Model Info"
                    )
                    reload_btn = gr.Button(
                        "🔄 Reload Model vào Engine", variant="primary"
                    )
                    reload_status = gr.Textbox(
                        label="Kết quả Reload", interactive=False, lines=2
                    )

                # ── Cột phải: upload model mới ─────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("#### 📤 Upload Model Mới")
                    model_file = gr.File(
                        label="Chọn file model (.pt)",
                        file_types=[".pt"],
                        file_count="single",
                    )
                    model_desc = gr.Textbox(
                        label="Mô tả (tùy chọn)",
                        placeholder="VD: Epoch 50 - val_loss 0.21 - acc 0.89",
                    )
                    with gr.Row():
                        model_epoch = gr.Textbox(label="Epoch", placeholder="50")
                        model_loss = gr.Textbox(label="Val Loss", placeholder="0.21")
                        model_acc = gr.Textbox(label="Accuracy", placeholder="0.89")

                    upload_model_btn = gr.Button(
                        "📥 Cập nhật Model", variant="primary"
                    )
                    upload_model_status = gr.Textbox(
                        label="Kết quả", interactive=False, lines=4
                    )

            gr.Markdown("---")
            gr.Markdown("#### 📋 Lịch sử Versions")

            versions_df = gr.Dataframe(
                value=get_versions_table(),
                label="Tất cả versions",
                interactive=False,
            )
            refresh_versions_btn = gr.Button("🔁 Làm mới danh sách")

            with gr.Row():
                rollback_input = gr.Textbox(
                    label="Rollback về version (VD: v1.0.0)",
                    placeholder="v1.0.0",
                    scale=3,
                )
                rollback_btn = gr.Button("⏪ Rollback", variant="stop", scale=1)

            rollback_status = gr.Textbox(
                label="Kết quả Rollback", interactive=False, lines=2
            )
            cleanup_btn = gr.Button("🗑️ Dọn dẹp backup cũ (giữ 3 bản gần nhất)")
            cleanup_status = gr.Textbox(
                label="Kết quả dọn dẹp", interactive=False
            )

            # ── Event handlers ──────────────────────────────────────────────
            upload_model_btn.click(
                handle_model_upload,
                inputs=[model_file, model_desc, model_epoch, model_loss, model_acc],
                outputs=[upload_model_status, versions_df],
            )
            reload_btn.click(
                handle_reload_model,
                outputs=[reload_status, model_info_md],
            )
            refresh_versions_btn.click(
                lambda: get_versions_table(), outputs=[versions_df]
            )
            rollback_btn.click(
                handle_rollback,
                inputs=[rollback_input],
                outputs=[rollback_status, versions_df],
            )
            cleanup_btn.click(handle_cleanup, outputs=[cleanup_status])


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False)
