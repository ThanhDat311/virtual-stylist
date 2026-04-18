import torch
import torch.nn as nn
import os
import numpy as np

# ── Màu sắc và color harmony ──────────────────────────────────────────────────
_NEUTRALS = {"black", "white", "grey", "gray", "silver", "brown", "beige", "cream", "khaki", "navy", "taupe"}
_COMPLEMENTARY = [
    ({"red", "pink", "maroon"}, {"green", "olive", "lime", "mint"}),
    ({"blue", "cyan", "teal", "navy"}, {"orange", "coral", "peach", "brown"}),
    ({"yellow", "gold"}, {"purple", "violet"}),
]
_WARM = {"red", "yellow", "orange", "pink", "maroon", "burgundy", "coral", "peach", "gold"}
_COOL = {"blue", "green", "teal", "cyan", "purple", "violet", "olive", "turquoise", "lime", "mint"}


def _color_harmony_bonus(color_a: str, color_b: str) -> float:
    """
    Tính điểm thưởng color harmony (0.0 ~ 0.15).
    Dùng để boost siamese score khi 2 màu phối tốt với nhau.
    """
    a = str(color_a).lower().strip() if color_a else ""
    b = str(color_b).lower().strip() if color_b else ""

    if not a or not b or a == "unknown" or b == "unknown":
        return 0.0

    # Monochromatic → tốt nhất
    if a == b:
        return 0.12

    # Neutral + anything → an toàn, luôn ổn
    if a in _NEUTRALS or b in _NEUTRALS:
        return 0.10

    # Complementary → điểm nhấn cao
    for p1, p2 in _COMPLEMENTARY:
        if (a in p1 and b in p2) or (a in p2 and b in p1):
            return 0.08

    # Warm analogous
    if a in _WARM and b in _WARM:
        return 0.05

    # Cool analogous
    if a in _COOL and b in _COOL:
        return 0.05

    # Clash (warm vs cool, unknown combo)
    return -0.05


# ── Model Architecture ────────────────────────────────────────────────────────
class CompatibilityMLP(nn.Module):
    def __init__(self, emb_dim=512, hidden=256):
        super().__init__()
        in_dim = emb_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )

    def forward(self, a, b):
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        return self.net(torch.cat([a, b, torch.abs(a - b)], dim=1)).squeeze(-1)


# ── Engine ────────────────────────────────────────────────────────────────────
class SiameseEngine:
    """
    Engine tính tương thích trang phục dựa trên Siamese MLP + color harmony.

    Features:
    - Tự động load model khi khởi tạo
    - Hỗ trợ hot-reload model mới không cần restart (gọi reload_model())
    - Khi model chưa load được → dùng color harmony làm fallback thông minh
    - Ensemble scoring: Siamese score + color harmony bonus
    """

    def __init__(self, model_path: str = "models/siamese_best.pt"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CompatibilityMLP | None = None
        self.model_loaded = False
        self._load_model()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load (hoặc reload) model từ file. Gọi lại khi cần hot-reload."""
        try:
            architecture = CompatibilityMLP(512, 256).to(self.device)
            # weights_only=False cho phép load checkpoint đầy đủ
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state", checkpoint)
            else:
                state_dict = checkpoint

            architecture.load_state_dict(state_dict)
            architecture.eval()

            self.model = architecture
            self.model_loaded = True
            print(f"[SiameseEngine] ✅ Model loaded from '{self.model_path}' on {self.device}")
        except FileNotFoundError:
            print(f"[SiameseEngine] ⚠️  Model not found at '{self.model_path}'. Using color-harmony fallback.")
            self.model_loaded = False
        except Exception as e:
            print(f"[SiameseEngine] ⚠️  Failed to load model: {e}. Using color-harmony fallback.")
            self.model_loaded = False

    def reload_model(self, new_path: str | None = None):
        """
        Hot-reload model mà KHÔNG cần restart app.

        Args:
            new_path: Nếu muốn đổi sang file khác. Nếu None → reload từ self.model_path.

        Returns:
            (success: bool, message: str)
        """
        if new_path:
            self.model_path = new_path
        prev_state = self.model_loaded
        self._load_model()

        if self.model_loaded:
            return True, f"✅ Đã reload model thành công từ '{self.model_path}'"
        elif prev_state:
            return False, f"❌ Reload thất bại. Đang dùng lại model cũ trong bộ nhớ."
        else:
            return False, f"❌ Reload thất bại và không có model cũ để fallback."

    # ── Scoring ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def rank_compatibility(
        self,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        query_color: str = "",
        candidate_colors: list[str] | None = None,
        color_weight: float = 0.25,
    ) -> np.ndarray:
        """
        Tính điểm tương thích giữa 1 item query và nhiều candidates.

        Args:
            query_emb: numpy (512,) — embedding của item đang chọn.
            candidate_embs: numpy (N, 512) — embeddings của các candidates.
            query_color: màu của item query (để tính color harmony bonus).
            candidate_colors: list màu của từng candidate (len == N).
            color_weight: trọng số của color harmony trong ensemble (0~1).
                          0.0 → chỉ dùng Siamese; 1.0 → chỉ dùng color.

        Returns:
            numpy (N,) — điểm tổng hợp, cao hơn = phù hợp hơn.
        """
        n = len(candidate_embs)

        # ── 1. Tính color harmony scores ──────────────────────────────────────
        if candidate_colors and len(candidate_colors) == n:
            harmony_bonus = np.array(
                [_color_harmony_bonus(query_color, c) for c in candidate_colors],
                dtype=np.float32,
            )
        else:
            harmony_bonus = np.zeros(n, dtype=np.float32)

        # ── 2. Tính Siamese model scores (nếu có model) ───────────────────────
        if self.model_loaded and self.model is not None:
            q = torch.tensor(query_emb, dtype=torch.float32).to(self.device)
            c = torch.tensor(candidate_embs, dtype=torch.float32).to(self.device)
            q_exp = q.expand(c.size(0), -1)
            siamese_scores = self.model(q_exp, c).cpu().numpy()  # (N,) in [0,1]
        else:
            # Fallback thông minh: cosine similarity thay vì random
            norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            norm_c = candidate_embs / (
                np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-9
            )
            cosine = np.dot(norm_c, norm_q)  # (N,)
            # Scale cosine từ [-1,1] sang [0,1]
            siamese_scores = (cosine + 1.0) / 2.0

        # ── 3. Ensemble: weighted combination ─────────────────────────────────
        ensemble = (1.0 - color_weight) * siamese_scores + color_weight * (
            harmony_bonus * 0.5 + 0.5  # map bonus (-0.15~0.15) → (0.42~0.57)
        )

        return ensemble.astype(np.float32)

    def get_model_status(self) -> dict:
        """Trả về thông tin trạng thái model cho UI."""
        return {
            "loaded": self.model_loaded,
            "path": self.model_path,
            "device": self.device,
            "file_exists": os.path.exists(self.model_path),
            "file_size_mb": (
                round(os.path.getsize(self.model_path) / 1024 / 1024, 2)
                if os.path.exists(self.model_path)
                else 0
            ),
        }
