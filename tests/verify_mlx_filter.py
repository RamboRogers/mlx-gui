#!/usr/bin/env python3
"""Verify MLX library filtering works correctly."""

import time
from huggingface_hub import list_models

def test_mlx_library_filter():
    """Test different filtering approaches."""
    
    print("🔍 Testing MLX Library Filter...")
    
    try:
        print("\n1️⃣ Using library='mlx' parameter:")
        models = list(list_models(library="mlx", limit=5, sort="downloads", direction=-1))
        for model in models:
            print(f"  - {model.id} (library: {getattr(model, 'library_name', 'unknown')})")
        
        time.sleep(2)  # Rate limit buffer
        
        print("\n2️⃣ Using tags=['mlx'] parameter:")
        models = list(list_models(tags=["mlx"], limit=5, sort="downloads", direction=-1))
        for model in models:
            print(f"  - {model.id} (tags: {getattr(model, 'tags', [])})")
        
        time.sleep(2)  # Rate limit buffer
        
        print("\n3️⃣ Checking specific known MLX models:")
        known_mlx_models = [
            "mlx-community/gemma-3n-E4B-it-bf16",
            "mlx-community/Qwen2-VL-2B-Instruct-4bit", 
            "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-MLX-4bit"
        ]
        
        for model_id in known_mlx_models:
            try:
                from huggingface_hub import model_info
                model = model_info(model_id)
                library = getattr(model, 'library_name', 'unknown')
                tags = getattr(model, 'tags', [])
                print(f"  - {model_id}")
                print(f"    Library: {library}")
                print(f"    MLX in tags: {'mlx' in [tag.lower() for tag in tags] if tags else False}")
                print(f"    Tags: {tags[:5]}...")  # First 5 tags
                print()
                time.sleep(1)
            except Exception as e:
                print(f"  - {model_id}: Error - {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mlx_library_filter()