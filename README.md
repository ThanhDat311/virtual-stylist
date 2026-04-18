# 👗 Virtual Stylist AI

Ứng dụng AI phối đồ cá nhân hóa — tự động gợi ý outfit từ tủ quần áo của bạn bằng mô hình **Siamese Neural Network** + **CLIP Vision Encoder**.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🧥 **Smart Closet** | Upload ảnh quần áo → AI tự phân loại vai trò (áo/quần/giày/phụ kiện), màu sắc, chất liệu |
| 👔 **AI Stylist** | Nhập yêu cầu tiếng Việt hoặc tiếng Anh → AI gợi ý outfit hoàn chỉnh |
| 🎨 **Color Harmony** | Điểm tương thích kết hợp Siamese score + Color Wheel Theory |
| ⚙️ **Model Manager** | Upload model mới từ Colab, hot-reload không cần restart, rollback version cũ |

---

## 🚀 Cài đặt và chạy

### 1. Clone repo
```bash
git clone https://github.com/<YOUR_USERNAME>/virtual-stylist.git
cd virtual-stylist
```

### 2. Cài dependencies
```bash
pip install -r requirements.txt
```

> **PyTorch:** Cài riêng theo CUDA version của máy bạn:
> ```bash
> # CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # CPU only
> pip install torch torchvision
> ```

### 3. Chạy app
```bash
python app.py
```

Mở trình duyệt tại `http://localhost:7860`

---

## 📁 Cấu trúc Project

```
virtual-stylist/
├── app.py                      # Gradio UI chính
├── update_model.py             # CLI cập nhật model
├── requirements.txt
├── models/
│   ├── siamese_best.pt         # Model weights (Siamese MLP)
│   └── registry.json           # Lịch sử versions
├── src/
│   ├── siamese_engine.py       # Scoring engine (Siamese + Color Harmony)
│   ├── model_manager.py        # Quản lý vòng đời model
│   ├── closet_manager.py       # Xử lý và lưu trữ tủ quần áo
│   ├── feature_extraction.py   # CLIP feature extractor
│   ├── styling_logic.py        # Rules phối đồ
│   ├── explainer.py            # Giải thích lý do gợi ý (XAI)
│   ├── style_classifier.py     # Phân loại phong cách (Casual/Office/Chic/Sport)
│   ├── prompt_processor.py     # Xử lý prompt song ngữ Việt-Anh
│   ├── color_classifier.py     # CLIP Zero-shot màu sắc
│   ├── sustainability.py       # Tính điểm bền vững
│   └── vector_db.py            # FAISS vector store
├── scripts/
│   ├── prepare_data.py         # Chuẩn bị dataset H&M
│   └── visualize_embeddings.py # T-SNE/PCA visualization
└── HM_Siamese_Compatibility.ipynb  # Notebook train model trên Colab
```

---

## 🔄 Cập nhật Model Mới từ Colab

Sau khi train xong trên Colab, có **2 cách** để update model:

### Cách 1 — Dùng UI (khuyến nghị)
1. Download file `.pt` từ Google Drive về máy
2. Vào tab **`⚙️ Model Manager`** trong app
3. Upload file → điền thông tin epoch/loss/accuracy
4. Nhấn **"📥 Cập nhật Model"** rồi **"🔄 Reload Model"**

### Cách 2 — Dùng CLI
```bash
python update_model.py path/to/new_model.pt --desc "Epoch 50" --epoch 50 --loss 0.21 --acc 0.89
```

### Export từ Colab
```python
torch.save({
    'model_state': model.state_dict(),
    'epoch': epoch,
    'val_loss': val_loss,
    'accuracy': accuracy,
}, 'siamese_best.pt')
```

---

## 🧠 Kiến trúc Model

```
CLIP Image Encoder (ViT-B/32, 512-dim)
           ↓
    [Embedding A]  [Embedding B]
           ↓
   Concat([A, B, |A-B|]) → 1536-dim
           ↓
    Linear(1536→256) + LayerNorm + GELU
    Linear(256→128) + LayerNorm + GELU
    Linear(128→1) + Sigmoid
           ↓
   Compatibility Score [0, 1]
```

**Ensemble Scoring:** `final_score = 0.75 × siamese_score + 0.25 × color_harmony_bonus`

---

## 📦 Dependencies chính

- `gradio` — Web UI
- `torch` — Deep Learning
- `transformers` — CLIP model
- `rembg` — Background removal
- `faiss-cpu` / `faiss-gpu` — Vector search
- `scikit-learn` — KMeans clustering
- `pandas`, `numpy`, `Pillow`

---

## 📄 License

MIT License
