##################################################pearson##################################################
#top_N1 = 150 , top_N2 = 2000은 유기적으로 설정 후 테스트 

# =========================
# Import Libraries
# =========================
import random
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from tqdm import tqdm
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
    'EPOCHS': 20,
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

# 라벨 데이터 스케일링 (선택사항)
# standard_scaler = StandardScaler()
# train_label_vec = standard_scaler.fit_transform(train_label_vec)
# val_label_vec = standard_scaler.transform(val_label_vec)

# =========================
# HEG and HVG Extraction Function
# =========================
def extract_HEG_HVG(expression_data, top_N1=150,top_N2=2000):
    gene_means = np.mean(expression_data, axis=0)
    gene_std = np.std(expression_data, axis=0)
    gene_cv = gene_std / (gene_means + 1e-8)

    HEG_indices = np.argsort(gene_means)[-top_N:]
    HVG_indices = np.argsort(gene_cv)[-top_N:]

    return HEG_indices, HVG_indices

# Extract HEG and HVG indices
top_N1 = 150
top_N2 = 2000
HEG_indices, HVG_indices = extract_HEG_HVG(train_label_vec, top_N1=top_N1,top_N2=top_N2)

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
        print("BaseModel Initialized")

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

# Pearson Loss 정의
class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, preds, targets):
        vx = preds - torch.mean(preds, dim=0, keepdim=True)
        vy = targets - torch.mean(targets, dim=0, keepdim=True)
        cost = torch.sum(vx * vy, dim=0) / (
            torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + 1e-8
        )
        loss = 1 - torch.mean(cost)
        return loss

# =========================
# Total Loss Function with Pearson Loss
# =========================
def total_loss_function(output, target, heg_weight=1.0, hvg_weight=1.0):
    huber_loss = nn.MSELoss()(output, target)
    pearson_loss = PearsonLoss()(output, target)

    heg_loss = nn.SmoothL1Loss()(output[:, HEG_indices], target[:, HEG_indices]) * heg_weight
    hvg_loss = nn.SmoothL1Loss()(output[:, HVG_indices], target[:, HVG_indices]) * hvg_weight

    total_loss = huber_loss
    return total_loss, huber_loss, heg_loss, hvg_loss, pearson_loss

# =========================
# Pearson Correlation Calculation for PCC Evaluation
# =========================
def calculate_pcc_score(preds, true_labels, heg_indices, hvg_indices):
    # 샘플별 상관계수 (MeanCorr_Cells)
    mean_corr_cells = np.mean([
        pearsonr(preds[i, :], true_labels[i, :])[0] for i in range(true_labels.shape[0])
    ])

    # 유전자별 최대 상관계수 (MaxCorr_Genes)
    max_corr_genes = np.max([
        pearsonr(preds[:, j], true_labels[:, j])[0] for j in range(true_labels.shape[1])
    ])

    # HEG 유전자 상관계수 평균
    heg_corr = np.mean([
        pearsonr(preds[:, idx], true_labels[:, idx])[0] for idx in heg_indices
    ])

    # HVG 유전자 상관계수 평균
    hvg_corr = np.mean([
        pearsonr(preds[:, idx], true_labels[:, idx])[0] for idx in hvg_indices
    ])

    pcc_score = max((mean_corr_cells + max_corr_genes + heg_corr + hvg_corr) / 4, 0)

    return pcc_score, mean_corr_cells, max_corr_genes, heg_corr, hvg_corr

# =========================
# Training Function with PCC Score Calculation
# =========================
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    best_pcc_score = -float('inf')  # Initialize best PCC Score
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            total_loss, huber_loss, heg_loss, hvg_loss, pearson_loss = total_loss_function(output, labels)

            total_loss.backward()
            optimizer.step()

            train_loss.append(total_loss.item())

            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        pcc_train, mean_corr_train, max_corr_train, heg_corr_train, hvg_corr_train = calculate_pcc_score(
            all_preds, all_labels, HEG_indices, HVG_indices
        )

        _val_loss, pcc_val = validation(model, val_loader, device)
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}], Val Loss : [{_val_loss:.5f}]')
        print(f'Train PCC Score: {pcc_train:.5f}, MeanCorr_Cells: {mean_corr_train:.5f}, MaxCorr_Genes: {max_corr_train:.5f}, '
              f'HEG_Corr: {heg_corr_train:.5f}, HVG_Corr: {hvg_corr_train:.5f}')
        print(f'Validation PCC Score: {pcc_val:.5f}')
        print(f'Huber Loss: {huber_loss.item():.5f}, HEG Loss: {heg_loss.item():.5f}, HVG Loss: {hvg_loss.item():.5f}, Pearson Loss: {pearson_loss.item():.5f}')

        if scheduler is not None:
            scheduler.step(_val_loss)

        # Save best model based on PCC score
        if pcc_val > best_pcc_score:
            best_pcc_score = pcc_val
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model

# =========================
# Validation Function with PCC Score Calculation
# =========================
def validation(model, val_loader, device):
    model.eval()
    val_loss = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss, huber_loss, heg_loss, hvg_loss, pearson_loss = total_loss_function(pred, labels)

            val_loss.append(loss.item())

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    _val_loss = np.mean(val_loss)

    pcc_val, mean_corr_val, max_corr_val, heg_corr_val, hvg_corr_val = calculate_pcc_score(
        all_preds, all_labels, HEG_indices, HVG_indices
    )

    print(f'Validation Huber Loss: {huber_loss.item():.5f}, HEG Loss: {heg_loss.item():.5f}, HVG Loss: {hvg_loss.item():.5f}, Pearson Loss: {pearson_loss.item():.5f}')
    print(f'Val PCC Score: {pcc_val:.5f}, MeanCorr_Cells: {mean_corr_val:.5f}, MaxCorr_Genes: {max_corr_val:.5f}, '
              f'HEG_Corr: { heg_corr_val:.5f}, HVG_Corr: {hvg_corr_val:.5f}')


    return _val_loss, pcc_val

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

# 필요한 경우 inverse_transform 적용
# preds = standard_scaler.inverse_transform(preds)

# 결과 저장 또는 제출 코드 추가

print("no scaler , only mse")
