import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from rembg import remove
from transformers import CLIPProcessor, CLIPModel

# Configuration
CLOSET_DIR = "user_closet"
CLOSET_IMG_DIR = os.path.join(CLOSET_DIR, "images")
CLOSET_EMB_DIR = os.path.join(CLOSET_DIR, "embeddings")
CSV_PATH = "user_closet_data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP Labels for 2-Layer Tagging
ROLE_LABELS = ["upper body clothing", "lower body clothing", "shoes", "fashion accessories"]
ROLE_MAPPING = {
    "upper body clothing": "UPPER",
    "lower body clothing": "LOWER",
    "shoes": "SHOES",
    "fashion accessories": "ACCESSORIES"
}

COLOR_LABELS = ["black", "white", "blue", "red", "green", "yellow", "brown", "gray", "pink", "purple", "orange", "beige"]
MATERIAL_LABELS = ["denim", "cotton", "leather", "silk", "wool", "polyester", "linen"]

class ClosetManager:
    def __init__(self):
        print(f"Initializing ClosetManager on {DEVICE}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if not os.path.exists(CSV_PATH):
            df = pd.DataFrame(columns=["id", "filename", "role", "color", "material", "embedding_path"])
            df.to_csv(CSV_PATH, index=False)
            
        os.makedirs(CLOSET_IMG_DIR, exist_ok=True)
        os.makedirs(CLOSET_EMB_DIR, exist_ok=True)

    def process_new_item(self, image_input):
        """
        Process a new image: remove background, classify, and save.
        image_input: PIL Image or path
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # 1. Background Removal
        print("Removing background...")
        img_no_bg = remove(img)
        # Convert RGBA to RGB (with white background for CLIP)
        background = Image.new("RGB", img_no_bg.size, (255, 255, 255))
        background.paste(img_no_bg, mask=img_no_bg.split()[3]) # 3 is alpha channel
        
        # 2. Extract Features & Tagging
        print("Auto-tagging with CLIP...")
        item_id = str(int(pd.Timestamp.now().timestamp() * 1000))
        filename = f"{item_id}.png"
        save_path = os.path.join(CLOSET_IMG_DIR, filename)
        img_no_bg.save(save_path) # Save original transparent one for UI

        # Tagging Layer 1: Role
        role = self._classify(background, ROLE_LABELS)
        mapped_role = ROLE_MAPPING.get(role, "UNKNOWN")
        
        # Tagging Layer 2: Visual attributes
        color = self._classify(background, COLOR_LABELS)
        material = self._classify(background, MATERIAL_LABELS)
        
        # 3. Save Embedding
        emb_filename = f"{item_id}.npy"
        emb_path = os.path.join(CLOSET_EMB_DIR, emb_filename)
        embedding = self._get_embedding(background)
        np.save(emb_path, embedding)

        # 4. Update Database
        new_row = {
            "id": item_id,
            "filename": filename,
            "role": mapped_role,
            "color": color,
            "material": material,
            "embedding_path": emb_path
        }
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
        return new_row

    def _classify(self, image, labels):
        inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        return labels[probs.argmax().item()]

    def _get_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_outputs = self.model.get_image_features(**inputs)
        
        # Ensure we have a tensor (some versions return an object)
        if not isinstance(vision_outputs, torch.Tensor):
            if hasattr(vision_outputs, "pooler_output"):
                vision_outputs = vision_outputs.pooler_output
            else:
                vision_outputs = vision_outputs[0]

        # Normalize
        embedding = vision_outputs / vision_outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def clear_closet(self):
        """Delete all items in the closet and reset the CSV."""
        import shutil
        if os.path.exists(CLOSET_DIR):
            try:
                shutil.rmtree(CLOSET_DIR)
            except Exception as e:
                print(f"Error deleting directory: {e}")
                
        os.makedirs(CLOSET_IMG_DIR, exist_ok=True)
        os.makedirs(CLOSET_EMB_DIR, exist_ok=True)
        
        df = pd.DataFrame(columns=["id", "filename", "role", "color", "material", "embedding_path"])
        df.to_csv(CSV_PATH, index=False)
        return "✅ Closet cleared successfully!"

if __name__ == "__main__":
    # Quick test
    # manager = ClosetManager()
    # manager.process_new_item("test_image.jpg")
    pass
