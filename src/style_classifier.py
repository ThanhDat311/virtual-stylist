"""
Style Classifier module for Virtual Stylist.
Phân loại phong cách trang phục (Casual, Chic, Office, Sport, v.v.)
Dựa trên rules (article_type, category) hoặc có thể mở rộng chạy CLIP Zero-shot sau này.
"""

class StyleClassifier:
    def __init__(self):
        # Định nghĩa các rules map từ article_type sang style
        self.style_rules = {
            "Office": [
                "shirts", "trousers", "formal shoes", "blazers", "suits", 
                "skirts", "formal shirts", "belts", "ties", "laptop bag"
            ],
            "Casual": [
                "tshirts", "jeans", "shorts", "sneakers", "flip flops", 
                "casual shoes", "sweatshirts", "jackets", "sweaters", "backpack"
            ],
            "Chic": [
                "dresses", "heels", "handbags", "jewellery", "clutches", 
                "sunglasses", "scarves", "flats", "kurtas", "tops"
            ],
            "Sport": [
                "track pants", "sports shoes", "tracksuits", "sports sandals", 
                "duffel bag", "caps", "water bottle"
            ]
        }
        
    def classify_style(self, item_metadata: dict) -> str:
        """
        Dự đoán phong cách của một món đồ dựa trên metadata.
        """
        article_type = str(item_metadata.get('article_type', '')).lower()
        category = str(item_metadata.get('category', '')).lower()
        
        # Rule-based matching
        for style, keywords in self.style_rules.items():
            if article_type in keywords:
                return style
                
        # Heuristics bổ sung nếu không match được article_type cụ thể
        if "sport" in article_type or "track" in article_type:
            return "Sport"
        if "formal" in article_type:
            return "Office"
        
        return "Casual" # Default fallback

    def extract_style_requirements(self, style: str) -> dict:
        """
        Trả về điều kiện lọc để dùng cho vector_db search khi User chọn một phong cách.
        Ví dụ: Chọn style Office -> Yêu cầu giày là 'formal shoes'.
        """
        style = style.capitalize()
        # Mặc định không nắn (để AI tự do mix)
        reqs = {
            "Footwear": [],
            "Bottomwear": []
        }
        
        if style == "Office":
            reqs["Footwear"] = ["formal shoes", "heels"]
            reqs["Bottomwear"] = ["trousers", "skirts"]
        elif style == "Casual":
            reqs["Footwear"] = ["casual shoes", "sneakers", "flip flops", "flats"]
            reqs["Bottomwear"] = ["jeans", "shorts", "tights"]
        elif style == "Chic":
            reqs["Footwear"] = ["heels", "flats", "sandals"]
            # Không ép bottomwear vì có thể là dresses
        elif style == "Sport":
            reqs["Footwear"] = ["sports shoes", "sports sandals"]
            reqs["Bottomwear"] = ["track pants", "shorts"]
            
        return reqs

def test():
    classifier = StyleClassifier()
    print("🧪 Testing Style Classifier...\n")
    
    test_cases = [
        {"name": "Nike Running Shoes", "article_type": "Sports Shoes"},
        {"name": "Allen Solly White Shirt", "article_type": "Shirts"},
        {"name": "Levis Blue Denim", "article_type": "Jeans"},
        {"name": "Prada Black Handbag", "article_type": "Handbags"},
        {"name": "Unknown Item", "article_type": "random"}
    ]
    
    for item in test_cases:
        style = classifier.classify_style(item)
        print(f"👕 Item: {item['name']} (Type: {item['article_type']})")
        print(f"👉 Predicted Style: {style}\n")
        
    print("📌 Style Requirements cho 'Office':")
    print(classifier.extract_style_requirements("Office"))

if __name__ == "__main__":
    test()
