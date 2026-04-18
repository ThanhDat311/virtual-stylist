import re

# Simple dictionary for fashion translation VN -> EN
VN_EN_MAPPING = {
    # Types
    "áo": "upper",
    "quần": "lower",
    "váy": "lower",
    "giày": "shoes",
    "dép": "shoes",
    "túi": "accessories",
    "mũ": "accessories",
    "nón": "accessories",
    "kính": "accessories",
    
    # Colors
    "đen": "black",
    "trắng": "white",
    "xanh": "blue",
    "đỏ": "red",
    "vàng": "yellow",
    "nâu": "brown",
    "xám": "gray",
    "hồng": "pink",
    "tím": "purple",
    "cam": "orange",
    "kem": "beige",
    
    # Materials
    "da": "leather",
    "bò": "denim",
    "vải": "cotton",
    "len": "wool",
}

def clean_text(text):
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def process_prompt(prompt):
    """
    Analyzes a bilingual prompt to extract search keywords.
    Returns: { 'role': str, 'color': str, 'material': str, 'keywords': list }
    """
    text = clean_text(prompt)
    words = text.split()
    
    extracted = {
        "role": None,
        "color": None,
        "material": None,
        "keywords": words
    }
    
    # Check for VN matches and convert to EN
    all_terms = []
    for word in words:
        if word in VN_EN_MAPPING:
            all_terms.append(VN_EN_MAPPING[word])
        else:
            all_terms.append(word)
            
    # Simple heuristic classification
    for term in all_terms:
        # Check Role
        if term in ["upper", "lower", "shoes", "accessories"]:
            extracted["role"] = term.upper()
        # Check Color (could be more than one, but we take first for now)
        elif term in ["black", "white", "blue", "red", "green", "yellow", "brown", "gray", "pink", "purple", "orange", "beige"]:
            if not extracted["color"]: extracted["color"] = term
        # Check Material
        elif term in ["denim", "cotton", "leather", "silk", "wool", "polyester", "linen"]:
            if not extracted["material"]: extracted["material"] = term
            
    return extracted

if __name__ == "__main__":
    # Test
    print(process_prompt("áo thun xanh bò"))
    print(process_prompt("blue denim shirt"))
