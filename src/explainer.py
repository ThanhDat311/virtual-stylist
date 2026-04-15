"""
Explainable AI (XAI) module for Virtual Stylist.
Cung cấp khả năng "giải thích" tại sao AI lại gợi ý một món đồ dựa trên 
Lý thuyết Bánh xe màu sắc (Color Wheel Theory) và quy tắc thời trang cơ bản.
"""

class OutfitExplainer:
    def __init__(self):
        # Định nghĩa các nhóm màu cơ bản
        self.neutrals = {"black", "white", "grey", "gray", "silver", "brown", "beige", "cream", "khaki", "navy", "taupe"}
        self.warm = {"red", "yellow", "orange", "pink", "maroon", "burgundy", "coral", "peach", "gold"}
        self.cool = {"blue", "green", "teal", "cyan", "purple", "violet", "olive", "turquoise", "lime", "mint"}
        
        # Các cặp màu tương phản bù sắc (Complementary)
        self.complementary_pairs = [
            ({"red", "pink", "maroon"}, {"green", "olive", "lime", "mint"}),
            ({"blue", "cyan", "teal", "navy"}, {"orange", "coral", "peach", "brown"}),
            ({"yellow", "gold"}, {"purple", "violet"})
        ]

    def _normalize_color(self, color_str):
        """Chuẩn hóa chuỗi màu sắc"""
        if not color_str:
            return "unknown"
        return str(color_str).lower().strip()

    def get_color_relationship(self, query_color: str, target_color: str) -> dict:
        """
        Phân tích mối quan hệ giữa 2 màu sắc dựa trên lý thuyết màu.
        Trả về dict chứa loại quan hệ và thông điệp giải thích.
        """
        qc = self._normalize_color(query_color)
        tc = self._normalize_color(target_color)

        if qc == "unknown" or tc == "unknown":
            return {"type": "Feature", "msg": "Phù hợp về kiểu dáng và phong cách tổng thể."}

        # 1. Monochromatic (Ton-sur-ton)
        if qc == tc or (qc in self.neutrals and tc in self.neutrals and qc != "white" and tc != "black" and qc != "black" and tc != "white"):
             if qc == tc:
                 return {"type": "Monochromatic", "msg": f"Phong cách Ton-sur-ton (cùng tone màu {tc}), tạo sự thanh lịch và liền mạch."}

        # 2. Neutral + Neutral (Black & White etc)
        if qc in self.neutrals and tc in self.neutrals:
            return {"type": "Neutral", "msg": "Sự kết hợp giữa các gam màu trung tính luôn mang lại sự sang trọng và an toàn."}

        # 3. Neutral + Colorful
        if (qc in self.neutrals and tc not in self.neutrals) or (qc not in self.neutrals and tc in self.neutrals):
            neutral_color = qc if qc in self.neutrals else tc
            pop_color = tc if qc in self.neutrals else qc
            return {"type": "Accent", "msg": f"Màu {neutral_color} trung tính làm nền tảng hoàn hảo, giúp tôn lên vẻ nổi bật của màu {pop_color}."}

        # 4. Complementary (Tương phản bù sắc)
        for pair1, pair2 in self.complementary_pairs:
            if (qc in pair1 and tc in pair2) or (qc in pair2 and tc in pair1):
                return {"type": "Complementary", "msg": f"Phối màu tương phản bù sắc (Complementary) giữa {qc} và {tc}, tạo điểm nhấn thị giác cực kỳ nổi bật và cá tính."}

        # 5. Warm/Cool Analagous (Tương đồng)
        if qc in self.warm and tc in self.warm:
             return {"type": "Warm Analogous", "msg": f"Sự kết hợp các tone màu ấm ({qc} và {tc}) mang lại cảm giác năng động và rực rỡ."}
        if qc in self.cool and tc in self.cool:
             return {"type": "Cool Analogous", "msg": f"Sự kết hợp các tone màu lạnh ({qc} và {tc}) tạo cảm giác mát mẻ, hài hòa và dịu mắt."}

        # Default fallback
        return {"type": "Harmonious", "msg": "Màu sắc hài hòa, kết hợp tốt với tổng thể trang phục."}

    def explain(self, query_item: dict, recommended_item: dict, match_reason: str = None) -> str:
        """
        Tạo câu giải thích tự nhiên hoàn chỉnh cho 1 món đồ được gợi ý.
        """
        q_color = query_item.get('color', 'unknown')
        r_color = recommended_item.get('color', 'unknown')
        r_subcat = recommended_item.get('sub_category', 'Món đồ')
        
        # Lấy quan hệ màu sắc
        rel = self.get_color_relationship(q_color, r_color)
        
        # Sinh câu văn
        explanation = f"Gợi ý {r_subcat.lower()} này vì: {rel['msg']}"
        
        if match_reason and "match" in match_reason.lower():
            # Tinh chỉnh câu chữ nếu có truyền match_reason (ví dụ: "Matched Footwear")
            pass # Giữ nguyên hoặc có thể thêm "Đảm nhiệm vai trò [Phụ kiện] trong outfit..."
            
        return explanation

def test():
    explainer = OutfitExplainer()
    print("🧪 Testing Explainable AI (Color Theory)...\n")
    
    test_cases = [
        ({"color": "Navy"}, {"color": "White", "sub_category": "Bottomwear"}),
        ({"color": "Red"}, {"color": "Green", "sub_category": "Topwear"}),
        ({"color": "Blue"}, {"color": "Orange", "sub_category": "Footwear"}),
        ({"color": "Black"}, {"color": "Black", "sub_category": "Accessories"}),
        ({"color": "Pink"}, {"color": "Red", "sub_category": "Topwear"}),
        ({"color": "Yellow"}, {"color": "Purple", "sub_category": "Bottomwear"})
    ]
    
    for q_item, r_item in test_cases:
        print(f"👕 Input: {q_item['color']}  +  👖 Gợi ý: {r_item['color']} ({r_item['sub_category']})")
        print(f"👉 Giải thích: {explainer.explain(q_item, r_item)}\n")

if __name__ == "__main__":
    test()
