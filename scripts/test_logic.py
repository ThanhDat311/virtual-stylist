import sys
import os
import json
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vector_db import VectorDatabase
from src.styling_logic import StyleRecommender

def main():
    print("=" * 50)
    print("🧪 Khởi chạy Test Pipeline - Giai đoạn 3")
    print("=" * 50)
    
    # 1. Khởi tạo FAISS
    db = VectorDatabase(index_path="data/vector_index.bin")
    
    # 2. Xử lý Metadata
    metadata_path = "data/metadata.json"
    if not os.path.exists(metadata_path):
        print("❌ Lỗi: Không tìm thấy data/metadata.json")
        return
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✅ Đã load {len(metadata)} metadata items.")
    
    # Init Style Recommender
    recommender = StyleRecommender(metadata=metadata)
    
    # 3. Lấy thử một ID là áo (Topwear) làm mẫu
    query_id = None
    query_item = None
    
    for k, v in metadata.items():
        if str(v.get('sub_category', '')).lower() == 'topwear':
            query_id = k
            query_item = v
            break
            
    if not query_id:
        print("⚠️ Không tìm thấy Topwear nào trong dataset.")
        return
        
    print(f"\n👕 [ITEM ĐẦU VÀO]: {query_item['name']}")
    print(f"   Category: {query_item['category']} > {query_item.get('sub_category')}")
    print(f"   Color: {query_item.get('color')}")
    
    # 4. Để test, lấy vector mẫu từ embeddings.npy (vì chúng ta không chạy CLIP real-time ở test script này)
    # File embeddings.npy nằm đâu?
    emb_path = "data/embeddings.npy"
    if not os.path.exists(emb_path):
        print("❌ Lỗi: Không tìm thấy embeddings.npy để test")
        return
        
    embeddings = np.load(emb_path)
    faiss_idx = query_item.get('faiss_index')
    if faiss_idx is None or faiss_idx >= len(embeddings):
        print("❌ Lỗi: faiss_index không hợp lệ.")
        return
        
    query_features = embeddings[faiss_idx]
    
    # 5. Chạy Mix & Match
    print("\n⏳ Đang Mix & Match Outfit ...")
    response = recommender.mix_and_match(
        query_features=query_features, 
        vector_db=db, 
        query_item_metadata=query_item,
        target_style="Casual"
    )
    
    outfit = response['outfit']
    style = response['style']
    sustainability = response['sustainability']
    
    print(f"\n✨ [KẾT QUẢ GỢI Ý OUTFIT ✨] - Style: {style}")
    print(f"🌿 Sustainability Score: {sustainability['score']}/100 ({sustainability['label']})")
    print(f"👉 Giải thích: {sustainability['explanation']}")
    print("-" * 40)
    
    for item_data in outfit:
        item = item_data['item']
        print(f"📌 {item_data['match_reason']}:")
        print(f"   - Tên: {item['name']}")
        print(f"   - Phân loại: {item['category']} > {item.get('sub_category')}")
        print(f"   - Màu sắc: {item.get('color')}")
        print(f"   - Độ khớp/Tương đồng: {item_data['similarity']:.4f}")
        print(f"   - 👗 AI Giải Thích: {item_data.get('explanation', 'Không có')}\n")

if __name__ == "__main__":
    main()
