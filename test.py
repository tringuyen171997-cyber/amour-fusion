import pickle
import pandas as pd

df = pd.read_csv("data/BTXRD/processed_metadata.csv") # Check your args.metadata_csv path
# Filter validation slice to replicate your validation check precisely
val_df = df[df['split'] == 'valid'].reset_index(drop=True)

with open("/root/amour-fusion/data/BTXRD/images_encoded.p", "rb") as f:
    img_feats = pickle.load(f)

print(f"Metadata rows: {len(val_df)} | Encoded embedding rows: {len(img_feats)}")