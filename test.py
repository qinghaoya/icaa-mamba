import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.models import create_model
from scipy.stats import spearmanr, pearsonr

import icaa_mamba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = create_model(
    'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
    pretrained=False,
    num_classes=1,
    img_size=224
)
model.to(device)


ckpt_path = "/data1/xqh/Vim-main/checkpoints/model.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

state_dict = checkpoint["model_state_dict"]

# DDP 兼容
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.eval()


normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


csv_file = "/data1/xqh/data/1test.csv"
image_root = "/data1/xqh/data"  

df = pd.read_csv(csv_file)

results = []

gt_list = []
pred_list = []

with torch.no_grad():
    for idx, row in df.iterrows():
        img_name = str(row.iloc[0])
        gt_score = float(row.iloc[1])

        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        img = default_loader(img_path)
        img = transform(img).unsqueeze(0).to(device)

        pred = model(img).item()

        gt_list.append(gt_score)
        pred_list.append(pred)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)}")

srcc, _ = spearmanr(gt_list, pred_list)
plcc, _ = pearsonr(gt_list, pred_list)

print(f"SRCC: {srcc:.3f}")
print(f"PLCC: {plcc:.3f}")
