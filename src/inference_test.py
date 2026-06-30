import torch
from src.modules import ExpModel # Assuming this is your LightningModule

# 1. Load the model from your specific checkpoint
checkpoint_path = "/root/amour-fusion/output/model/both/bone_class/1979/epoch=16-valid_score=1.000.ckpt"
model = ExpModel.load_from_checkpoint(checkpoint_path)
model.eval()

def infer_text_only(txt_embedding_tensor):
    batch_size = txt_embedding_tensor.shape[0]
    
    # Create missing image input (zeros) and mask (0)
    img_input = torch.zeros(batch_size, 3584)
    img_mask = torch.zeros(batch_size, 1, dtype=torch.long)
    
    # Text present
    txt_mask = torch.ones(batch_size, 1, dtype=torch.long)
    
    with torch.no_grad():
        # Adjust arguments to match your model's forward signature
        logits = model(img_input, txt_embedding_tensor, None, img_mask, txt_mask)
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)