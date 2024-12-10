##############앙상블 모델 feature concat##############



# =========================
# Import Libraries
# =========================
import random
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, StandardScaler

import warnings
warnings.filterwarnings(action='ignore')

# =========================
# Device Configuration
# =========================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# =========================
# Hyperparameter Setting
# =========================
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 10,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 32,
    'SEED': 41
}

# =========================
# Fixed Random Seed
# =========================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# =========================
# Data Pre-processing
# =========================
df = pd.read_csv('./train.csv')

# Train-Validation Split (random 0.8:0.2 split)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CFG['SEED'])

# Reset index after splitting
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Extract Labels
train_label_vec = train_df.iloc[:, 2:].values.astype(np.float32)
val_label_vec = val_df.iloc[:, 2:].values.astype(np.float32)

CFG['label_size'] = train_label_vec.shape[1]

# =========================
# HEG and HVG Extraction Function
# =========================
def extract_HEG_HVG(expression_data, top_N=100):
    gene_means = np.mean(expression_data, axis=0)
    gene_std = np.std(expression_data, axis=0)
    gene_cv = gene_std / (gene_means + 1e-8)

    HEG_indices = np.argsort(gene_means)[-top_N:]
    HVG_indices = np.argsort(gene_cv)[-top_N:]

    return HEG_indices, HVG_indices

# Extract HEG and HVG indices
top_N = 5
HEG_indices, HVG_indices = extract_HEG_HVG(train_label_vec, top_N=top_N)

# =========================
# Custom Dataset
# =========================
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list=None, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if self.label_list is not None:
            label = self.label_list[index]
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

# =========================
# Data Transformations
# =========================
train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# =========================
# Create Datasets and Loaders
# =========================
train_dataset = CustomDataset(train_df['path'].values, train_label_vec, train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_df['path'].values, val_label_vec, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# =========================
# Model Definition
# =========================
class BaseModel(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b1(pretrained=True)
        self.regressor = nn.Linear(1000, gene_size)
        print("base")


    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

class BaseModel2(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel2, self).__init__()
        self.d_backbone = models.efficientnet_b1(pretrained=True)
        self.s_backbone = models.googlenet(pretrained=True)

        # ViT 모델 로드 (ImageNet-21k로 사전 학습된 가중치 사용)
        self.v_backbone = timm.create_model('timm/vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        # 분류 헤드를 제거하고 특징 추출기만 사용
        self.v_backbone.head = nn.Identity()
        # 백본의 출력 특징 수
        self.num_features = self.v_backbone.num_features

        # 차원 축소 레이어
        self.d_down = nn.Linear(1000, self.num_features)
        self.regressor = nn.Linear(self.num_features * 2, gene_size)  # 두 개의 백본 출력 특징을 결합하므로 2배

        print("base2")

    def forward(self, x):
        # d_backbone의 출력
        d = self.d_backbone(x)  # (batch_size, 1000)
        d = self.d_down(d)      # (batch_size, num_features)

        # v_backbone의 출력
        v = self.v_backbone(x)  # (batch_size, num_features)

        # d와 v의 특징을 결합 (Concatenation)
        combined_features = torch.cat((d, v), dim=1)  # (batch_size, num_features * 2)

        # Regressor 적용
        x = self.regressor(combined_features)  # (batch_size, gene_size)
        return x


class BaseModel3(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel2, self).__init__()
        self.d_backbone = models.efficientnet_b1(pretrained=True)
        self.s_backbone = models.googlenet(pretrained=True)

        # ViT 모델 로드 (ImageNet-21k로 사전 학습된 가중치 사용)
        self.v_backbone = timm.create_model('timm/vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        # 분류 헤드를 제거하고 특징 추출기만 사용
        self.v_backbone.head = nn.Identity()
        # 백본의 출력 특징 수
        self.num_features = self.v_backbone.num_features

        # 차원 축소 레이어
        self.d_down = nn.Linear(1000, self.num_features)
        self.regressor = nn.Linear(self.num_features , gene_size)  # 두 개의 백본 출력 특징을 결합하므로 2배

        print("base3")

    def forward(self, x):
        # d_backbone의 출력
        d = self.d_backbone(x)  # (batch_size, 1000)
        d = self.d_down(d)      # (batch_size, num_features)

        # v_backbone의 출력
        v = self.v_backbone(x)  # (batch_size, num_features)

        # d와 v의 특징을 결합 (Concatenation)
        combined_features = torch.cat((d, v), dim=1)  # (batch_size, num_features * 2)

        # Regressor 적용
        x = self.regressor(v)  # (batch_size, gene_size)
        return x

# =========================
# Total Loss Function with Weighted HEG/HVG Genes
# =========================
def total_loss_function(output, target, heg_weight=5.0, hvg_weight=2.0):
    # 기본 Huber 손실 계산
    huber_loss = nn.SmoothL1Loss()(output, target)

    # HEG/HVG 유전자에 가중치를 적용한 손실 계산
    heg_loss = nn.SmoothL1Loss()(output[:, HEG_indices], target[:, HEG_indices]) * heg_weight
    hvg_loss = nn.SmoothL1Loss()(output[:, HVG_indices], target[:, HVG_indices]) * hvg_weight

    # 총 손실
    total_loss = huber_loss + heg_loss + hvg_loss
    return total_loss, huber_loss, heg_loss, hvg_loss

# =========================
# Training Function with Huber Loss and HEG/HVG Weights
# =========================
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    best_loss = float('inf')
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            total_loss, huber_loss, heg_loss, hvg_loss = total_loss_function(output, labels)

            total_loss.backward()
            optimizer.step()

            train_loss.append(total_loss.item())

        _val_loss = validation(model, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_loss)

        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model

# =========================
# Validation Function
# =========================
def validation(model, val_loader, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss, _, _, _ = total_loss_function(pred, labels)

            val_loss.append(loss.item())

    _val_loss = np.mean(val_loss)
    return _val_loss

# =========================
# Inference Function
# =========================
def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            pred = model(imgs)

            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds

# =========================
# Run Training
# =========================
model = BaseModel()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                       threshold_mode='abs', min_lr=1e-8, verbose=True)

trained_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

# =========================
# Inference and Submission
# =========================
test_df = pd.read_csv('./test.csv')

# Create Test Dataset and Loader
test_dataset = CustomDataset(test_df['path'].values, label_list=None, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# Perform Inference
preds = inference(trained_model, test_loader, device)
