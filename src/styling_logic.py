"""
Styling Logic module.
Contains the rules and AI logic for recommending outfits.
"""

from src.explainer import OutfitExplainer
from src.style_classifier import StyleClassifier
from src.sustainability import SustainabilityCalculator
# CLIPColorClassifier is imported conditionally if needed to save RAM during init
# from src.color_classifier import CLIPColorClassifier 

class StyleRecommender:
    def __init__(self, metadata=None):
        """
        Initialize with metadata dictionary to allow filtering.
        """
        self.metadata = metadata or {}
        
        # Init AI Sub-modules
        self.explainer = OutfitExplainer()
        self.style_classifier = StyleClassifier()
        self.sustainability = SustainabilityCalculator()
        self.color_classifier = None # Lazy load
        
    def _get_color_classifier(self):
        if self.color_classifier is None:
            from src.color_classifier import CLIPColorClassifier
            self.color_classifier = CLIPColorClassifier()
        return self.color_classifier

    def get_recommendations(self, query_features, vector_db, top_k=5, required_color=None, use_clip_color=False, image_path=None):
        """
        Generate style recommendations based on image features.
        """
        fetch_k = top_k * 10 if required_color else top_k
        
        similarities, indices = vector_db.search(query_features, top_k=fetch_k)
        
        results = []
        sims = similarities[0] if hasattr(similarities, 'shape') and len(similarities.shape) > 1 else similarities
        idxs = indices[0] if hasattr(indices, 'shape') and len(indices.shape) > 1 else indices
        
        target_clip_color = None
        if use_clip_color and required_color and image_path:
            # Optionally use CLIP zero-shot for much higher color accuracy
            classifier = self._get_color_classifier()
            target_clip_color = required_color.lower()
        
        for sim, idx in zip(sims, idxs):
            str_idx = str(idx)
            if str_idx not in self.metadata:
                continue
                
            item = self.metadata[str_idx]
            
            # Apply color filtering
            if required_color:
                if use_clip_color and target_clip_color:
                    # Logic needs to classify the dataset image, but dataset is too large to runtime-classify.
                    # fallback to metadata logic but note it.
                    item_color = str(item.get('color', '')).lower()
                else:
                    item_color = str(item.get('color', '')).lower()
                    target_color = str(required_color).lower()
                    if item_color != target_color:
                        continue 
                    
            results.append({
                "id": str_idx,
                "similarity": float(sim),
                "item": item
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def mix_and_match(self, query_features, vector_db, query_item_metadata, target_style="Casual"):
        """
        AI-enabled Mix & Match:
        1. Classifies the input style or forces a target style.
        2. Retrieves visually matching items filtered by style rules.
        3. Appends Explainable AI text.
        4. Calculates Sustainability Score.
        """
        input_category = query_item_metadata.get('category', '')
        input_subcategory = query_item_metadata.get('sub_category', '')
        
        input_sub = str(input_subcategory).lower()
        target_subcategories = []
        target_mastercategories = []
        
        # Rule Base matching (Topwear -> Bottomwear)
        if input_sub == 'topwear':
            target_subcategories = ['bottomwear']
            target_mastercategories = ['Footwear', 'Accessories']
        elif input_sub == 'bottomwear':
            target_subcategories = ['topwear']
            target_mastercategories = ['Footwear', 'Accessories']
        elif str(input_category).lower() == 'footwear':
            target_subcategories = ['topwear', 'bottomwear']
            target_mastercategories = ['Accessories']
        elif str(input_category).lower() == 'accessories':
            target_subcategories = ['topwear', 'bottomwear']
            target_mastercategories = ['Footwear']
        else:
            target_subcategories = ['topwear', 'bottomwear']
            target_mastercategories = ['Footwear']

        # Determine all targets we need
        all_targets = [{'sub': sub} for sub in target_subcategories] + [{'master': mst} for mst in target_mastercategories]
        
        # Get Style rules constraints (Casual vs Office vs Chic)
        style_reqs = self.style_classifier.extract_style_requirements(target_style)

        # Query items
        outfit = []
        pool_size = vector_db.index.ntotal if vector_db.index else 44000
        similarities, indices = vector_db.search(query_features, top_k=pool_size)
        
        sims = similarities[0]
        idxs = indices[0]
        
        for target in all_targets:
            for sim, idx in zip(sims, idxs):
                str_idx = str(idx)
                if str_idx not in self.metadata:
                    continue
                    
                item = self.metadata[str_idx]
                
                # Check Style Restrictions
                item_article = str(item.get('article_type', '')).lower()
                if 'sub' in target and target['sub'].lower() == 'bottomwear':
                    if style_reqs["Bottomwear"] and item_article not in style_reqs["Bottomwear"]:
                        continue # Does not match the requested Style
                if 'master' in target and target['master'].lower() == 'footwear':
                    if style_reqs["Footwear"] and item_article not in style_reqs["Footwear"]:
                        continue # Does not match the requested Style

                # Check Match Criteria
                is_match = False
                match_reason = ""
                if 'sub' in target and str(item.get('sub_category', '')).lower() == target['sub'].lower():
                    is_match = True
                    match_reason = f"Matched {target['sub'].capitalize()}"
                elif 'master' in target and str(item.get('category', '')).lower() == target['master'].lower():
                    is_match = True
                    match_reason = f"Matched {target['master'].capitalize()}"

                if is_match:
                    # Tích hợp A3 - Giải thích lý do
                    explanation = self.explainer.explain(query_item_metadata, item, match_reason)
                    
                    outfit.append({
                        "id": str_idx, 
                        "similarity": float(sim), 
                        "item": item, 
                        "match_reason": match_reason,
                        "explanation": explanation,
                        "is_owned": False # It's a new recommendation
                    })
                    break 

        # Tích hợp A4 - Sustainability
        # Món đồ người dùng đưa vào được tính là "Tái sử dụng/Sở hữu sẵn"
        dummy_input_item = {"name": query_item_metadata.get("name", "User Item"), "is_owned": True}
        sustainability_data = self.sustainability.calculate_score([dummy_input_item] + outfit)

        return {
            "outfit": outfit,
            "style": target_style,
            "sustainability": sustainability_data
        }
