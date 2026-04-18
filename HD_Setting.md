## � Yêu cầu hệ thống đã tích hợp UI/UX

### Phần mềm cần thiết:
- **Python**: Phiên bản 3.8 trở lên (khuyên dùng 3.10+)
- **Git**: Để clone repository (tùy chọn)
- **VS Code**: Với extension Python, Pylance (khuyên dùng)

### Thư viện Python (tự động cài từ requirements.txt):
- torch (PyTorch)
- transformers
- faiss-cpu
- gradio
- pandas
- Pillow
- rembg[cpu]
- kagglehub[pandas-datasets]
- scikit-learn
- matplotlib
- seaborn
- tqdm
- numpy

---

## 🚀 Hướng dẫn cài đặt và chạy chi tiết

### Bước 1: Chuẩn bị môi trường

1. **Cài đặt Python**:
   - Tải từ [python.org](https://www.python.org/downloads/)
   - Đảm bảo thêm Python vào PATH khi cài đặt
   - Kiểm tra phiên bản: Mở Command Prompt/Terminal và chạy:
     ```bash
     python --version
     ```
     Nên hiển thị Python 3.8+ (ví dụ: Python 3.14.2)

2. **Cài đặt VS Code** (tùy chọn nhưng khuyên dùng):
   - Tải từ [code.visualstudio.com](https://code.visualstudio.com/)
   - Cài đặt extension cần thiết:
     - Python (ms-python.python)
     - Pylance (ms-python.vscode-pylance)
     - Python Debugger (ms-python.debugpy)

### Bước 2: Tải và chuẩn bị project

1. **Clone hoặc tải project**:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/virtual-stylist.git
   cd virtual-stylist
   ```
   Hoặc tải ZIP từ GitHub và giải nén.

2. **Mở project trong VS Code**:
   - Mở VS Code
   - File > Open Folder > Chọn thư mục virtual-stylist

### Bước 3: Cài đặt dependencies

1. **Mở Terminal trong VS Code**:
   - View > Terminal
   - Hoặc Ctrl + ` (backtick)

2. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Lưu ý cho Windows**: Nếu gặp lỗi "pip is not recognized", hãy dùng:
   > ```bash
   > python -m pip install -r requirements.txt
   > ```

3. **Cài đặt thêm rembg với CPU support** (nếu chưa có):
   ```bash
   pip install "rembg[cpu]"
   ```

4. **Kiểm tra cài đặt**:
   - Chạy lệnh sau để kiểm tra các thư viện chính:
     ```bash
     python -c "import torch, transformers, gradio, pandas, PIL; print('Tất cả thư viện đã cài đặt thành công!')"
     ```
     Nếu không có lỗi, mọi thứ đã sẵn sàng.

### Bước 4: Chạy ứng dụng

1. **Chạy app**:
   ```bash
   python app.py
   ```
   
   > **Tự động mở trình duyệt**: App sẽ tự động mở trình duyệt tại `http://127.0.0.1:7861`. Nếu không mở được, truy cập thủ công.

2. **Mở trình duyệt**:
   - App sẽ chạy trên `http://127.0.0.1:7861` (hoặc `http://localhost:7861`)
   - Mở trình duyệt web và truy cập địa chỉ trên
   - Giao diện Gradio sẽ hiển thị với các tab: Smart Closet, AI Stylist, Color Harmony, Model Manager

3. **Sử dụng app**:
   - **Smart Closet**: Upload ảnh quần áo để AI phân loại
   - **AI Stylist**: Nhập mô tả outfit mong muốn (ví dụ: "áo sơ mi trắng với quần jean xanh")
   - **Color Harmony**: Xem điểm tương thích màu sắc
   - **Model Manager**: Quản lý các phiên bản model AI

### Bước 5: Khắc phục sự cố

- **Lỗi "Module not found"**: Chạy lại `pip install -r requirements.txt`
- **Lỗi PyTorch**: Cài PyTorch phù hợp với CUDA version:
  ```bash
  # CPU only
  pip install torch torchvision
  # CUDA 11.8
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **App không mở trong trình duyệt**: Kiểm tra firewall hoặc thử địa chỉ `http://127.0.0.1:7860`
- **Lỗi rembg**: Đảm bảo đã cài `pip install "rembg[cpu]"`

---

## 📁 Cấu trúc Project

```
virtual-stylist/
├── app.py                 # File chính chạy ứng dụng
├── requirements.txt       # Danh sách thư viện cần thiết
├── README.md             # Hướng dẫn này
├── models/               # Thư mục chứa model AI
│   ├── registry.json
│   └── siamese_best.pt
├── scripts/              # Scripts tiện ích
│   ├── prepare_data.py
│   ├── test_logic.py
│   └── visualize_embeddings.py
└── src/                  # Source code
    ├── closet_manager.py
    ├── color_classifier.py
    ├── explainer.py
    ├── feature_extraction.py
    ├── model_manager.py
    ├── prompt_processor.py
    ├── siamese_engine.py
    ├── style_classifier.py
    ├── styling_logic.py
    ├── sustainability.py
    └── vector_db.py
```

---

## 🤝 Đóng góp

Chào mừng đóng góp! Tạo issue hoặc pull request trên GitHub.

---

## 📄 Giấy phép

MIT License

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