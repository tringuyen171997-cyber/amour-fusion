import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Encode text and images using Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--metadata_csv", type=str, default="data/bone_disease_mock/metadata.csv")
    parser.add_argument("--output_notes", type=str, default="data/BTXRD/notes_encoded.p")
    parser.add_argument("--output_images", type=str, default="data/BTXRD/images_encoded.p")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--use_mock", action="store_true", help="Force use of mock random embeddings")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load metadata
    if not os.path.exists(args.metadata_csv):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_csv}")
    
    df = pd.read_csv(args.metadata_csv)
    print(f"Loaded {len(df)} samples from {args.metadata_csv}")
    
    model = None
    processor = None
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    if not args.use_mock:
        try:
            print(f"Attempting to load foundation model: {args.model_id} on {device}...")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(args.model_id)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map=None
            )
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load model {args.model_id} due to: {e}")
            print("Falling back to random mock feature extractor (1536 dimensions).")
            model = None
            processor = None

    notes_data = []
    images_data = []
    
    base_dir = os.path.dirname(args.metadata_csv)
    
    for idx, row in df.iterrows():
        hadm_id = row["HADM_ID"]
        note_text = row["clinical_note"]
        img_rel_path = row["image_path"]
        
        # 1. Encode text
        if model is not None and processor is not None:
            try:
                text_inputs = processor(text=[note_text], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    outputs = model(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    # Average pool text tokens to get a single note embedding
                    last_hidden = outputs.hidden_states[-1][0] # shape [seq_len, hidden_dim]
                    text_emb = last_hidden.mean(dim=0).cpu().numpy().tolist()
            except Exception as ex:
                print(f"Error encoding text for HADM_ID {hadm_id}: {ex}. Using random mockup.")
                text_emb = np.random.randn(1536).tolist()
        else:
            text_emb = np.random.randn(1536).tolist()
            
        notes_data.append({
            "HADM_ID": hadm_id,
            "vector": text_emb
        })
        
        # 2. Encode image (if present)
        img_emb = None
        if isinstance(img_rel_path, str) and img_rel_path != "":
            img_abs_path = os.path.join(base_dir, img_rel_path)
            if os.path.exists(img_abs_path):
                if model is not None and processor is not None:
                    try:
                        img = Image.open(img_abs_path).convert("RGB")
                        inputs = processor(images=img, return_tensors="pt")
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        with torch.no_grad():
                            visual_outputs = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
                            img_emb = visual_outputs.mean(dim=0).cpu().numpy()
                    except Exception as ex:
                        print(f"Error encoding image {img_abs_path}: {ex}. Using random mockup.")
                        img_emb = np.random.randn(1536)
                else:
                    img_emb = np.random.randn(1536)
            else:
                print(f"Warning: Image file not found: {img_abs_path}")
        
        images_data.append({
            "HADM_ID": hadm_id,
            "vector": img_emb
        })
        
        if (idx + 1) % 5 == 0:
            print(f"Encoded {idx + 1}/{len(df)} samples...")
            
    # Save encoded pickle dataframes
    notes_df = pd.DataFrame(notes_data)
    images_df = pd.DataFrame(images_data)
    
    with open(args.output_notes, "wb") as f:
        pickle.dump(notes_df, f)
        
    with open(args.output_images, "wb") as f:
        pickle.dump(images_df, f)
        
    print(f"Encoded text saved to {args.output_notes}")
    print(f"Encoded images saved to {args.output_images}")

if __name__ == "__main__":
    main()
