"""
Sustainability module for Virtual Stylist.
Tính toán điểm số "Thời trang bền vững" (Sustainability Score).
Khuyến khích người dùng tận dụng lại đồ cũ (mix & match) thay vì mua sắm mới.
"""

class SustainabilityCalculator:
    def __init__(self):
        # Trung bình lượng khí thải/nước tiết kiệm được khi không mua 1 món đồ mới
        self.CO2_SAVED_PER_ITEM_KG = 2.5
        self.WATER_SAVED_PER_ITEM_LITERS = 2000

    def calculate_score(self, outfit_items: list) -> dict:
        """
        Tính toán các chỉ số môi trường dựa trên số lượng đồ tái sử dụng.
        
        Args:
            outfit_items: Danh sách các dict món đồ. Mỗi item có thể có cờ 'is_owned' (boolean)
                          báo hiệu đây là đồ user có sẵn trong tủ, thay vì đồ mua mới.
        """
        if not outfit_items:
            return {
                "score": 0,
                "co2_saved_kg": 0,
                "water_saved_liters": 0,
                "reuse_rate": "0/0",
                "label": "Chưa có thông tin"
            }

        total_items = len(outfit_items)
        # Giả định: Item đầu vào dùng làm tham chiếu ban đầu luôn là đồ có sẵn (is_owned = True)
        reused_count = sum(1 for item in outfit_items if item.get("is_owned", False))

        # Điểm /100
        score = int((reused_count / total_items) * 100) if total_items > 0 else 0

        # Môi trường
        co2_saved = reused_count * self.CO2_SAVED_PER_ITEM_KG
        water_saved = reused_count * self.WATER_SAVED_PER_ITEM_LITERS

        # Gắn nhãn
        if score == 100:
            label = "🌍 Nữ Hoàng Bền Vững (100% Tái sử dụng)"
            explanation = "Thật tuyệt vời! Bạn đã ghép được một bộ đồ hoàn chỉnh mà không cần mua thêm món mới nào."
        elif score >= 50:
            label = "🌿 Thân Thiện Với Môi Trường"
            explanation = "Bạn đang đi đúng hướng! Tận dụng lại đồ cũ giúp hạn chế rác thải thời trang."
        elif score > 0:
            label = "🌱 Bước Đầu Bền Vững"
            explanation = "Một món đồ được tái sử dụng là một bước bảo vệ môi trường."
        else:
            label = "⚠️ Fast Fashion Warning"
            explanation = "Mua sắm thông minh nhé! Hãy thử mix đồ mới này với những món trong tủ đồ của bạn."

        return {
            "score": score,
            "co2_saved_kg": round(co2_saved, 1),
            "water_saved_liters": water_saved,
            "reuse_rate": f"{reused_count}/{total_items} items reused",
            "label": label,
            "explanation": explanation
        }

def test():
    calculator = SustainabilityCalculator()
    print("🧪 Testing Sustainability Calculator...\n")
    
    # Kịch bản 1: Mix & Match với 1 áo có sẵn, mua thêm 3 món mới
    outfit_1 = [
        {"name": "Áo cũ của tôi", "is_owned": True},
        {"name": "Quần mới gợi ý", "is_owned": False},
        {"name": "Giày mới gợi ý", "is_owned": False},
        {"name": "Mũ mới gợi ý", "is_owned": False}
    ]
    
    # Kịch bản 2: Mix & Match với 3 món có sẵn trong tủ
    outfit_2 = [
        {"name": "Áo cũ", "is_owned": True},
        {"name": "Quần cũ", "is_owned": True},
        {"name": "Giày cũ", "is_owned": True},
        {"name": "Kính mới", "is_owned": False}
    ]

    print("--- Kịch bản 1: Mới mua sắm ---")
    res1 = calculator.calculate_score(outfit_1)
    for k, v in res1.items():
        print(f"  {k}: {v}")

    print("\n--- Kịch bản 2: Tận dụng đồ cũ triệt để ---")
    res2 = calculator.calculate_score(outfit_2)
    for k, v in res2.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    test()
