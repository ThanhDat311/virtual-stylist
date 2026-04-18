import os
import gradio as gr
import pandas as pd
import numpy as np
from src.closet_manager import ClosetManager, CLOSET_IMG_DIR, CSV_PATH
from src.prompt_processor import process_prompt
from src.siamese_engine import SiameseEngine
from src.model_manager import ModelManager
from src.sustainability import SustainabilityCalculator

# ── Khởi tạo core modules ──────────────────────────────────────────────────────
closet_mgr = ClosetManager()
stylist_ai = SiameseEngine(model_path='models/siamese_best.pt')
model_mgr = ModelManager()
sustainability_calc = SustainabilityCalculator()


def _format_image_src(path: str) -> str:
    if not path:
        return ''
    abs_path = os.path.abspath(path)
    return f"file:///{abs_path.replace(os.sep, '/')}"


def load_closet():
    if not os.path.exists(CSV_PATH):
        return []
    try:
        df = pd.read_csv(CSV_PATH)
        return [
            (
                os.path.join(CLOSET_IMG_DIR, row.filename),
                f'{row.role} | {row.color} | {row.material}',
            )
            for _, row in df.iterrows()
        ]
    except Exception:
        return []


def get_closet_summary_html() -> str:
    if not os.path.exists(CSV_PATH):
        return '<div class="sidebar-card"><h3>Closet Summary</h3><p>Chưa có dữ liệu.</p></div>'

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    if total == 0:
        return '<div class="sidebar-card"><h3>Closet Summary</h3><p>Tủ đồ đang trống. Upload ngay để bắt đầu.</p></div>'

    roles = df['role'].value_counts().to_dict()
    colors = df['color'].value_counts().head(5).to_dict()
    materials = df['material'].value_counts().head(5).to_dict()

    role_chips = ''.join([f'<span class="chip">{role}: {count}</span>' for role, count in roles.items()])
    color_chips = ''.join([f'<span class="chip color-chip">{color}</span>' for color in colors])
    material_chips = ''.join([f'<span class="chip">{material}</span>' for material in materials])

    return f'''
    <div class="sidebar-card">
      <h3>Closet Summary</h3>
      <p><strong>{total}</strong> items in closet</p>
      <div class="chip-row">{role_chips}</div>
      <h4>Popular Colors</h4>
      <div class="chip-row">{color_chips}</div>
      <h4>Materials</h4>
      <div class="chip-row">{material_chips}</div>
    </div>
    '''


def get_quick_suggestions_html() -> str:
    if not os.path.exists(CSV_PATH):
        return '<div class="sidebar-card"><h3>Quick Suggestions</h3><p>Upload items để nhận gợi ý Outfit nhanh.</p></div>'

    df = pd.read_csv(CSV_PATH)
    if len(df) < 3:
        return '<div class="sidebar-card"><h3>Quick Suggestions</h3><p>Upload ít nhất 3 món đồ để tạo outfit.</p></div>'

    upper = df[df['role'] == 'UPPER'].head(1)
    lower = df[df['role'] == 'LOWER'].head(1)
    shoes = df[df['role'] == 'SHOES'].head(1)
    accessories = df[df['role'] == 'ACCESSORIES'].head(1)

    lines = []
    if not upper.empty and not lower.empty and not shoes.empty:
        lines.append(f'Áo {upper.iloc[0].color} + Quần {lower.iloc[0].color} + Giày {shoes.iloc[0].color}')
    if not upper.empty and not lower.empty and not accessories.empty:
        lines.append(f'Áo {upper.iloc[0].color} + Quần {lower.iloc[0].color} + Phụ kiện {accessories.iloc[0].color}')
    if not lower.empty and not shoes.empty and not accessories.empty:
        lines.append(f'Quần {lower.iloc[0].color} + Giày {shoes.iloc[0].color} + Phụ kiện {accessories.iloc[0].color}')

    if not lines:
        lines = ['Thêm nhiều món hơn để AI gợi ý outfit linh hoạt.']

    suggestions = ''.join([f'<li>{line}</li>' for line in lines[:3]])
    return f'''
    <div class="sidebar-card">
      <h3>Quick Outfit Suggestions</h3>
      <ul>{suggestions}</ul>
    </div>
    '''


def guess_style_tag(prompt_text: str) -> str:
    text = str(prompt_text or '').lower()
    if any(keyword in text for keyword in ['office', 'công sở', 'formal', 'chuyên nghiệp']):
        return 'Office'
    if any(keyword in text for keyword in ['sport', 'thể thao', 'gym', 'yoga']):
        return 'Sport'
    if any(keyword in text for keyword in ['chic', 'sang', 'party', 'đi chơi']):
        return 'Chic'
    return 'Casual'


def build_outfit_card_html(outfit: dict, index: int) -> str:
    item_blocks = []
    for item in outfit['items']:
        src = _format_image_src(item['path'])
        item_blocks.append(f'''
            <div class="item-chip">
              <div class="item-preview">
                <img src="{src}" alt="{item['role']}" loading="lazy" />
              </div>
              <div class="item-meta">
                <strong>{item['role']}</strong>
                <span>{item['color']} · {item['material']}</span>
              </div>
            </div>
            ''')

    sustainability = outfit['sustainability']
    return f'''
    <div class="outfit-card">
      <div class="outfit-card-header">
        <div>
          <span class="badge style-badge">{outfit['style']}</span>
          <h3>Look #{index}</h3>
        </div>
        <div class="score-pill">{outfit['score']}%</div>
      </div>
      <div class="outfit-card-grid">{''.join(item_blocks)}</div>
      <div class="outfit-card-details">
        <span class="badge harmony-badge">{outfit['color_harmony']}</span>
        <span class="badge sustainability-badge">Sustainability {sustainability['score']}%</span>
      </div>
      <div class="outfit-card-footer">
        <div class="small-metric">💚 {sustainability['label']}</div>
        <button class="try-on-btn">Try on</button>
      </div>
    </div>
    '''


def build_color_harmony_html() -> str:
    return '''
    <div class="glass-panel section-panel">
      <h2>Color Harmony</h2>
      <p>Powered by Siamese Neural Network + Color Wheel Theory, hệ thống gợi ý outfit cân bằng giữa tông màu, tương phản và sự hài hòa.</p>
      <div class="harmony-grid">
        <div class="harmony-box">
          <h4>Monochrome</h4>
          <p>Tông màu giống nhau giúp outfit sạch và high-fashion.</p>
        </div>
        <div class="harmony-box">
          <h4>Complementary</h4>
          <p>Màu đối lập tạo điểm nhấn bắt mắt, phù hợp phong cách Gen Z.</p>
        </div>
        <div class="harmony-box">
          <h4>Analogous</h4>
          <p>Nhóm màu liền kề cho cảm giác nhẹ nhàng, dễ phối.</p>
        </div>
      </div>
      <div class="color-wheel">
        <span class="wheel-dot dot-1"></span>
        <span class="wheel-dot dot-2"></span>
        <span class="wheel-dot dot-3"></span>
        <span class="wheel-dot dot-4"></span>
      </div>
    </div>
    '''


def create_outfit_collections(prompt_text: str):
    if not os.path.exists(CSV_PATH):
        return None, 'Closet is empty. Upload your fashion pieces first.'

    df = pd.read_csv(CSV_PATH)
    if len(df) == 0:
        return None, 'Closet is empty. Upload your fashion pieces first.'

    intent = process_prompt(prompt_text)
    filtered_df = df.copy()
    if intent['role']:
        filtered_df = df[df['role'] == intent['role']]
    if intent['color']:
        filtered_df = filtered_df[filtered_df['color'] == intent['color']]
    if intent['material']:
        filtered_df = filtered_df[filtered_df['material'] == intent['material']]

    if len(filtered_df) == 0:
        return None, f"Không tìm thấy món đồ nào phù hợp với: '{prompt_text}'"

    starters = filtered_df.head(4)
    outfits = []
    style_tag = guess_style_tag(prompt_text)

    for _, starter_item in starters.iterrows():
        starter_emb = np.load(starter_item.embedding_path)
        outfit_items = [
            {
                'path': os.path.join(CLOSET_IMG_DIR, starter_item.filename),
                'role': starter_item.role,
                'color': starter_item.color,
                'material': starter_item.material,
                'is_owned': True,
            }
        ]

        target_roles = ['UPPER', 'LOWER', 'SHOES', 'ACCESSORIES']
        if starter_item.role in target_roles:
            target_roles.remove(starter_item.role)

        scores = []
        unique_colors = {starter_item.color.lower()} if starter_item.color else set()
        for role in target_roles:
            candidates = df[df['role'] == role]
            if len(candidates) == 0:
                continue
            candidate_embs = np.stack([np.load(path) for path in candidates.embedding_path])
            candidate_colors = candidates['color'].tolist()
            role_scores = stylist_ai.rank_compatibility(
                query_emb=starter_emb,
                candidate_embs=candidate_embs,
                query_color=str(starter_item.color),
                candidate_colors=candidate_colors,
                color_weight=0.25,
            )
            best_idx = int(np.argmax(role_scores))
            best_item = candidates.iloc[best_idx]
            scores.append(float(role_scores[best_idx]))
            if best_item.color:
                unique_colors.add(best_item.color.lower())
            outfit_items.append(
                {
                    'path': os.path.join(CLOSET_IMG_DIR, best_item.filename),
                    'role': best_item.role,
                    'color': best_item.color,
                    'material': best_item.material,
                    'is_owned': True,
                }
            )

        if not scores:
            continue

        avg_score = int(np.mean(scores) * 100)
        if len(unique_colors) == 1:
            harmony_label = 'Monochrome Harmony'
        elif len(unique_colors) >= 3:
            harmony_label = 'Playful Contrast'
        else:
            harmony_label = 'Balanced Palette'

        sustainability = sustainability_calc.calculate_score(outfit_items)
        outfits.append(
            {
                'items': outfit_items,
                'score': avg_score,
                'style': style_tag,
                'color_harmony': harmony_label,
                'sustainability': sustainability,
            }
        )

    if not outfits:
        return None, 'Không đủ món đồ để tạo outfit hoàn chỉnh. Hãy upload thêm item.'

    cards = ''.join([build_outfit_card_html(outfit, idx + 1) for idx, outfit in enumerate(outfits[:4])])
    wrapper = f'<div class="outfit-grid">{cards}</div>'
    return wrapper, f'Đã tạo {min(len(outfits), 4)} outfit hoàn chỉnh.'


def clear_closet_action():
    msg = closet_mgr.clear_closet()
    return load_closet(), msg, get_closet_summary_html(), get_quick_suggestions_html()


def upload_and_refresh(images):
    if images is None:
        return load_closet(), 'Please select images.', get_closet_summary_html(), get_quick_suggestions_html()

    results = []
    for img in images:
        metadata = closet_mgr.process_new_item(img)
        results.append(f"Uploaded: {metadata['role']} ({metadata['color']})")

    return load_closet(), '\n'.join(results), get_closet_summary_html(), get_quick_suggestions_html()


def handle_style_request(prompt_text):
    card_html, msg = create_outfit_collections(prompt_text)
    if not card_html:
        return '<div class="empty-state">Không có outfit nào để hiển thị.</div>', msg, get_closet_summary_html(), get_quick_suggestions_html()
    return card_html, msg, get_closet_summary_html(), get_quick_suggestions_html()


# ── Tab 3: Model Manager ───────────────────────────────────────────────────────
def get_model_info_display():
    'Trả về string hiển thị thông tin model hiện tại.'
    info = model_mgr.get_current_model_info()
    status = stylist_ai.get_model_status()

    lines = [
        '## 🤖 Thông tin Model Hiện Tại',
        '',
        f"**Version:** `{info.get('version', 'N/A')}`",
        f"**Mô tả:** {info.get('description', 'N/A')}",
        f"**Ngày tạo:** {info.get('created_at', 'N/A')[:19].replace('T', ' ')}`",
        f"**Kích thước file:** {info.get('file_size_mb', 0)} MB",
        f"**MD5:** `{info.get('md5', 'N/A')[:12]}...`",
        '',
        '---',
        f"**Engine status:** {'✅ Model đã load' if status['loaded'] else '⚠️ Đang dùng fallback (cosine similarity + color harmony)'}`",
        f"**Device:** `{status['device'].upper()}`",
    ]

    ti = info.get('training_info', {})
    if ti:
        lines += [
            '',
            '**Training Info:**',
            *[f"  - {k}: {v}" for k, v in ti.items()],
        ]

    return '\n'.join(lines)


def get_versions_table():
    versions = model_mgr.list_versions()
    if not versions:
        return pd.DataFrame(columns=['Version', 'Ngày tạo', 'Size (MB)', 'Mô tả'])

    active = model_mgr.registry.get('active_version', '')
    rows = []
    for v in versions:
        rows.append(
            {
                'Version': ('⭐ ' if v['version'] == active else '') + v['version'],
                'Ngày tạo': v.get('created_at', '')[:19].replace('T', ' '),
                'Size (MB)': v.get('file_size_mb', 0),
                'Mô tả': v.get('description', ''),
            }
        )
    return pd.DataFrame(rows)


def handle_model_upload(file_obj, description_text, epoch_text, loss_text, acc_text):
    if file_obj is None:
        return '⚠️ Chưa chọn file.', get_versions_table()

    training_info = {}
    if epoch_text.strip():
        training_info['epoch'] = epoch_text.strip()
    if loss_text.strip():
        training_info['val_loss'] = loss_text.strip()
    if acc_text.strip():
        training_info['accuracy'] = acc_text.strip()

    src_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    success, msg = model_mgr.update_model(src_path, description=description_text, training_info=training_info)
    return msg, get_versions_table()


def handle_reload_model():
    success, msg = stylist_ai.reload_model()
    from src.model_manager import _load_registry
    model_mgr.registry = _load_registry()
    return msg, get_model_info_display()


def handle_rollback(version_str):
    version_str = version_str.strip().replace('⭐ ', '')
    if not version_str:
        return '⚠️ Chưa nhập version.', get_versions_table()
    success, msg = model_mgr.rollback(version_str)
    return msg, get_versions_table()


def handle_cleanup():
    msg = model_mgr.delete_old_backups(keep_last=3)
    return msg


# ── UI Construction ───────────────────────────────────────────────────────────
custom_css = '''
html, body {
  background: radial-gradient(circle at top left, #00142d 0%, #06070f 25%, #070a17 100%);
  color: #eef2ff;
}
.gradio-container {
  background: transparent !important;
}
#main-layout {
  display: flex;
  flex-direction: row-reverse;
}
.gradio-tabs .tab-button.selected {
  background: rgba(0, 255, 159, 0.14) !important;
  border-color: #00ff9f !important;
}
.gradio-tabs .tab-button {
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  backdrop-filter: blur(18px) !important;
}
.glass-panel,
.sidebar-card,
.outfit-card,
.version-card {
  background: rgba(10, 14, 32, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 18px 60px rgba(0, 255, 159, 0.08);
  backdrop-filter: blur(24px);
  border-radius: 24px;
}
.glass-panel {
  padding: 24px;
}
.sidebar-card {
  padding: 20px;
  margin-bottom: 18px;
}
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.chip {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  color: #f8fafc;
  font-size: 0.9rem;
  border: 1px solid rgba(255, 255, 255, 0.14);
}
.color-chip {
  background: linear-gradient(135deg, #00ff9f 0%, #ff2e9a 100%);
  box-shadow: 0 0 18px rgba(0, 255, 159, 0.18);
}
.outfit-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}
.outfit-card {
  padding: 20px;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.14);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.outfit-card:hover {
  transform: translateY(-6px) scale(1.01);
  box-shadow: 0 26px 90px rgba(0, 240, 255, 0.18);
}
.outfit-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.score-pill {
  padding: 8px 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #00ff9f, #00f0ff, #ff2e9a);
  color: #020617;
  font-weight: 700;
}
.outfit-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.item-chip {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 18px;
  padding: 12px;
  background: rgba(4, 8, 25, 0.7);
}
.item-preview {
  width: 100%;
  min-height: 120px;
  background: linear-gradient(145deg, rgba(0,255,159,0.12), rgba(255,46,154,0.08));
  border-radius: 18px;
  display: grid;
  place-items: center;
  overflow: hidden;
}
.item-preview img {
  width: 100%;
  height: auto;
  object-fit: cover;
  border-radius: 18px;
}
.item-meta {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.outfit-card-details {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 18px;
}
.badge {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.12);
}
.style-badge { background: rgba(0, 240, 255, 0.12); }
.harmony-badge { background: rgba(255, 46, 154, 0.12); }
.sustainability-badge { background: rgba(0, 255, 159, 0.12); }
.try-on-btn {
  border: none;
  padding: 12px 22px;
  border-radius: 999px;
  font-weight: 700;
  cursor: pointer;
  background: linear-gradient(135deg, #00ff9f, #ff2e9a);
  color: #030d16;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.try-on-btn:hover {
  transform: scale(1.04);
  box-shadow: 0 18px 30px rgba(255, 46, 154, 0.24);
}
.outfit-card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}
.small-metric {
  font-size: 0.95rem;
  color: #dbeafe;
}
.section-panel {
  margin-top: 20px;
}
.harmony-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 20px;
}
.harmony-box {
  padding: 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.12);
}
.color-wheel {
  position: relative;
  width: 240px;
  height: 240px;
  margin: 30px auto 0 auto;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,255,159,0.16) 0%, rgba(0,240,255,0.07) 40%, rgba(255,46,154,0.03) 100%);
  box-shadow: inset 0 0 40px rgba(0, 255, 159, 0.08);
}
.wheel-dot {
  position: absolute;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 0 20px rgba(0, 255, 159, 0.35);
}
.dot-1 { top: 20%; left: 50%; transform: translateX(-50%); }
.dot-2 { top: 50%; right: 18%; transform: translateY(-50%); }
.dot-3 { bottom: 18%; left: 50%; transform: translateX(-50%); }
.dot-4 { top: 50%; left: 18%; transform: translateY(-50%); }
'''

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue='indigo'),
    title='👗 Virtual Stylist AI',
    css=custom_css,
) as demo:
    gr.Markdown(
        '''
        # 👗 Virtual Stylist AI
        *Liquid Glassmorphism 2.0 · Gen Z Virtual Fashion Stylist*
        '''
    )

    with gr.Row(elem_id='main-layout'):
        with gr.Column(scale=1):
            sidebar_summary = gr.HTML(value=get_closet_summary_html())
            quick_suggestions = gr.HTML(value=get_quick_suggestions_html())

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab('🏠 Smart Closet'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            uploader = gr.File(
                                label='Drag & drop ảnh quần áo',
                                file_count='multiple',
                                file_types=['image'],
                            )
                            upload_btn = gr.Button('📥 Thêm vào Closet', variant='primary')
                            clear_btn = gr.Button('🗑️ Xóa toàn bộ Closet', variant='stop')
                            status_log = gr.Textbox(label='Trạng thái', interactive=False, lines=3)
                        with gr.Column(scale=2):
                            closet_gallery = gr.Gallery(
                                label='Tủ Quần Áo',
                                value=load_closet(),
                                columns=4,
                                height='620px',
                            )

                    upload_btn.click(
                        upload_and_refresh,
                        inputs=[uploader],
                        outputs=[closet_gallery, status_log, sidebar_summary, quick_suggestions],
                    )
                    clear_btn.click(
                        clear_closet_action,
                        outputs=[closet_gallery, status_log, sidebar_summary, quick_suggestions],
                    )

                with gr.Tab('✨ AI Stylist'):
                    gr.Markdown('### Phối đồ ngay với prompt song ngữ Việt + English')
                    with gr.Row():
                        prompt_input = gr.Textbox(
                            placeholder="VD: 'Phối cho tôi áo thun đen' hoặc 'I want an outfit with blue jeans'...",
                            label='Nhập yêu cầu của bạn',
                            lines=2,
                        )
                        style_btn = gr.Button('Phối đồ ngay', variant='primary')
                    output_msg = gr.Markdown('*Nhập yêu cầu rồi nhấn nút Phối đồ ngay!*')
                    outfit_cards = gr.HTML(value='<div class="empty-state">Chưa có outfit nào.</div>')

                    style_btn.click(
                        handle_style_request,
                        inputs=[prompt_input],
                        outputs=[outfit_cards, output_msg, sidebar_summary, quick_suggestions],
                    )

                with gr.Tab('🎨 Color Harmony'):
                    gr.HTML(value=build_color_harmony_html())

                with gr.Tab('⚙️ Model Manager'):
                    gr.Markdown(
                        '''
                        ### Quản lý Model & Hot-reload
                        Upload model mới `.pt`, xem phiên bản và rollback khi cần.
                        '''
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            model_info_md = gr.Markdown(value=get_model_info_display())
                            reload_btn = gr.Button('🔄 Reload Model vào Engine', variant='primary')
                            reload_status = gr.Textbox(label='Kết quả Reload', interactive=False, lines=3)
                        with gr.Column(scale=1):
                            model_file = gr.File(label='Chọn file model (.pt)', file_types=['.pt'], file_count='single')
                            model_desc = gr.Textbox(label='Mô tả (tùy chọn)', placeholder='Epoch 50 - val_loss 0.21 - acc 0.89')
                            with gr.Row():
                                model_epoch = gr.Textbox(label='Epoch', placeholder='50')
                                model_loss = gr.Textbox(label='Val Loss', placeholder='0.21')
                                model_acc = gr.Textbox(label='Accuracy', placeholder='0.89')
                            upload_model_btn = gr.Button('📥 Cập nhật Model', variant='primary')
                            upload_model_status = gr.Textbox(label='Kết quả', interactive=False, lines=4)
                    gr.Markdown('---')
                    versions_df = gr.Dataframe(value=get_versions_table(), interactive=False, label='Lịch sử Versions')
                    refresh_versions_btn = gr.Button('🔁 Làm mới danh sách')
                    with gr.Row():
                        rollback_input = gr.Textbox(label='Rollback về version', placeholder='v1.0.0')
                        rollback_btn = gr.Button('⏪ Rollback', variant='stop')
                    rollback_status = gr.Textbox(label='Kết quả Rollback', interactive=False, lines=2)
                    cleanup_btn = gr.Button('🗑️ Dọn dẹp backup cũ', variant='secondary')
                    cleanup_status = gr.Textbox(label='Kết quả dọn dẹp', interactive=False)

                    upload_model_btn.click(
                        handle_model_upload,
                        inputs=[model_file, model_desc, model_epoch, model_loss, model_acc],
                        outputs=[upload_model_status, versions_df],
                    )
                    reload_btn.click(handle_reload_model, outputs=[reload_status, model_info_md])
                    refresh_versions_btn.click(lambda: get_versions_table(), outputs=[versions_df])
                    rollback_btn.click(handle_rollback, inputs=[rollback_input], outputs=[rollback_status, versions_df])
                    cleanup_btn.click(handle_cleanup, outputs=[cleanup_status])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    demo.launch(share=False, inbrowser=True)
