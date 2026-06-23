import os
import pandas as pd
import numpy as np
from PIL import Image

def main():
    base_dir = "data/bone_disease_mock"
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    
    # Generate mock images
    # We will generate 30 mock samples. 
    # Some will have images, some will have missing images (image_path = "")
    data = []
    
    notes = [
        "Bệnh nhân nam, 45 tuổi, đau khớp gối dữ dội sau chấn thương thể thao. Phim chụp X-quang cho thấy khe khớp hẹp nhẹ.",
        "Nữ bệnh nhân, 60 tuổi, tiền sử loãng xương. Đau vùng lưng dưới lan xuống hông. Nghi ngờ gãy xẹp đốt sống.",
        "Trẻ em nam, 10 tuổi, ngã xe đạp, đau chói vùng cẳng tay phải. Sưng nề nhiều, hạn chế vận động.",
        "Bệnh nhân nữ, 32 tuổi, đau âm ỉ khớp háng bên trái khi đi lại. Vận động khớp háng bình thường.",
        "Bệnh nhân nam, 70 tuổi, thoái hóa khớp gối độ 3. Đau nhiều khi lên xuống cầu thang, có tiếng lục khục.",
    ]
    
    splits = ['train'] * 20 + ['val'] * 5 + ['test'] * 5
    
    for i in range(30):
        subject_id = 1000 + i
        hadm_id = 2000 + i
        
        # Determine if image is missing
        # Every 3rd case is missing the image (missing modality test)
        has_image = (i % 3 != 0)
        
        if has_image:
            image_name = f"img_{subject_id}.jpg"
            image_path = os.path.join("images", image_name)
            # Create a simple dummy image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(base_dir, image_path))
        else:
            image_path = ""
            
        note = notes[i % len(notes)] + f" (Mã số: {subject_id})"
        label = i % 3 # 3 classes: 0, 1, 2
        
        data.append({
            "SUBJECT_ID": subject_id,
            "HADM_ID": hadm_id,
            "ICUSTAY_ID": hadm_id,
            "clinical_note": note,
            "image_path": image_path,
            "label": label,
            "split": splits[i]
        })
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "metadata.csv"), index=False)
    print("Mock dataset generated successfully at:", base_dir)

if __name__ == "__main__":
    main()
