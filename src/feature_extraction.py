"""
Feature Extraction module.
Handles image processing and feature extraction using CLIP vision model.
Model: openai/clip-vit-base-patch32 (512-dim, normalized embeddings)
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import torch


class FeatureExtractor:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        """
        Load CLIP vision encoder. Falls back to CPU if GPU is not available.
        The model is lazy-loaded on first use to keep import fast.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load CLIP model on first call."""
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            print(f"[FeatureExtractor] Loading CLIP model on {self.device}...")
            self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            print("[FeatureExtractor] CLIP model loaded ✅")

    def process_image(self, image):
        """
        Preprocess an image before feature extraction:
          1. Remove background using rembg (if available), paste on white bg.
          2. Resize to 224×224 pixels.

        Args:
            image: PIL.Image, numpy array, or str (file path).
        Returns:
            PIL.Image (RGB, 224×224)
        """
        # --- Normalise input type ---
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        # --- Background removal (best-effort) ---
        try:
            from rembg import remove as rembg_remove
            result = rembg_remove(image)          # returns RGBA
            if result.mode == "RGBA":
                white_bg = Image.new("RGB", result.size, (255, 255, 255))
                white_bg.paste(result, mask=result.split()[3])
                image = white_bg
            else:
                image = result.convert("RGB")
        except Exception:
            pass  # rembg not installed or failed → use original

        # --- Resize to 224×224 ---
        image = image.resize((224, 224), Image.LANCZOS)
        return image

    def extract_features(self, image):
        """
        Extract a 512-dim L2-normalised feature vector using CLIP image encoder.

        Args:
            image: PIL.Image, numpy array, or str (file path).
        Returns:
            numpy.ndarray of shape (512,) — unit-norm vector for cosine similarity.
        """
        self._load_model()

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        # L2-normalise so cosine distance == dot product distance
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()  # shape: (512,)

    def extract_features_batch(self, image_paths, batch_size=32, verbose=True):
        """
        Extract features for a list of image file paths.

        Args:
            image_paths  : list[str] — file paths to images.
            batch_size   : int — number of images per CLIP forward pass.
            verbose      : bool — print progress every 10 batches.
        Returns:
            numpy.ndarray of shape (N, 512).
        """
        self._load_model()
        all_features = []
        total = len(image_paths)

        for batch_start in range(0, total, batch_size):
            batch_paths = image_paths[batch_start: batch_start + batch_size]
            batch_images, loaded_count = [], 0

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    loaded_count += 1
                except Exception as e:
                    if verbose:
                        print(f"  [skip] {path}: {e}")

            if not batch_images:
                continue

            inputs = self._processor(
                images=batch_images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                features = self._model.get_image_features(**inputs)

            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

            processed = min(batch_start + batch_size, total)
            if verbose and (batch_start // batch_size) % 10 == 0:
                print(f"  [FeatureExtractor] {processed}/{total} images done")

        if not all_features:
            return np.empty((0, 512), dtype=np.float32)

        return np.vstack(all_features).astype(np.float32)


class ColorExtractor:
    def __init__(self, num_colors=3):
        """
        Initialize the color extractor.
        Sử dụng phân cụm K-Means để trích xuất màu sắc.
        """
        self.num_colors = num_colors
        # Bảng màu tham chiếu (đơn giản, có thể mở rộng thêm)
        self.color_map = {
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Silver/Gray": (128, 128, 128),
            "Navy": (0, 0, 128),
            "Brown": (165, 42, 42),
            "Orange": (255, 165, 0),
            "Pink": (255, 192, 203)
        }

    def _closest_color_name(self, rgb_color):
        """Tính khoảng cách Euclidean từ màu trích được đến bảng màu chuẩn."""
        min_dist = float('inf')
        closest_name = "Unknown"
        for name, truth_rgb in self.color_map.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb_color, truth_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

    def extract_dominant_color(self, image_path_or_pil):
        """
        Trích xuất màu chủ đạo thực sự của chiếc giày, phớt lờ phông nền trắng.
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")
            
        # 1. CROP VÙNG TRUNG TÂM: Bỏ qua các rìa ảnh (thường chứa phông nền trắng)
        # Lấy một ô vuông ở giữa ảnh kích thước 50%
        width, height = image.size
        left = width * 0.25
        top = height * 0.25
        right = width * 0.75
        bottom = height * 0.75
        image = image.crop((left, top, right, bottom))
        
        # 2. TIỀN XỬ LÝ NHANH: Thu nhỏ ảnh lại còn 30x30 pixels để K-Means chạy trong mili-giây
        image = image.resize((30, 30))
        pixels = np.array(image).reshape(-1, 3)
        
        # 3. K-MEANS CLUSTERING: Tìm ra các nhóm màu phổ biến nhất
        kmeans = KMeans(n_clusters=self.num_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        counts = Counter(kmeans.labels_)
        
        # 4. CHỌN MÀU ĐẠI DIỆN: Loại bỏ cụm màu phông nền
        best_rgb = None
        # duyệt theo cụm màu xuất hiện nhiều nhất xuống ít nhất
        for cluster_idx, count in counts.most_common():
            rgb = kmeans.cluster_centers_[cluster_idx]
            # Mẹo: Ảnh thời trang thường có nền trắng (hoặc rất sáng). 
            # Nếu giá trị trung bình màu sáng > 240, xác suất cao nó là pixel bị dính nền.
            if np.mean(rgb) > 240:
                continue
            # Chốt màu thực sự đầu tiên tìm thấy
            best_rgb = rgb
            break
            
        # Fallback: Trắng toàn tập (Nếu thuật toán đã loại hết, chứng tỏ giày đó cũng full Trắng)
        if best_rgb is None:
            best_rgb = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
            
        color_name = self._closest_color_name(best_rgb)
        return color_name, best_rgb.astype(int).tolist()
