import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = torch.sigmoid(output)  # Sigmoid 적용
        return output

    # def forward(self, input):
    #     x0_0 = self.conv0_0(input)
    #     x1_0 = self.conv1_0(self.pool(x0_0))
    #     x2_0 = self.conv2_0(self.pool(x1_0))
    #     x3_0 = self.conv3_0(self.pool(x2_0))
    #     x4_0 = self.conv4_0(self.pool(x3_0))

    #     x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
    #     x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
    #     x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
    #     x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

    #     output = self.final(x0_4)
    #     return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

# # 데이터셋 클래스 정의
# class SegmentationDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         # 이미지 파일의 확장자를 필터링하여 파일 목록을 가져옴
#         self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(image_dir, f))]
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, img_name)
        
#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')
#         # print(np.array(image))
#         # print(np.array(mask))
#         plt.imshow(np.array(image)) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
#         # plt.imshow(np.array(mask)) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
#         plt.show() ## 이미지 인터프린터에 출력
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
        
#         return np.array(image), np.array(mask)


# # 데이터 로드 및 전처리
# def load_data(image_dir, mask_dir):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
    
#     dataset = SegmentationDataset(image_dir, mask_dir, transform=None)
    
#     # 빈 데이터셋 체크
#     if len(dataset) == 0:
#         raise ValueError("Dataset is empty. Check your image and mask directories.")
    
#     train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
#     return train_loader, test_loader


# 데이터셋 클래스 정의
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 이미지 파일의 확장자를 필터링하여 파일 목록을 가져옴
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(image_dir, f))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# 데이터 로드 및 전처리 (validation 추가)
def load_data(image_dir, mask_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 이미지를 고정 크기로 리사이즈
        transforms.ToTensor(),
    ])
    
    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    
    # 빈 데이터셋 체크
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your image and mask directories.")
    
    # train, validation, test 데이터로 나눔
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)  # 시드 고정
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 시드 고정 (train 60%, val 20%, test 20%)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


# IoU 계산 함수
def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# 훈련 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(iter(train_loader)):
        images, masks = images.to(device, torch.float32), masks.to(device, torch.float32)
        # print(images)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 새로운 validation 평가 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in tqdm(iter(val_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += iou_score(outputs, masks)
    
    return total_loss / len(val_loader), total_iou / len(val_loader)

# 평가 함수
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += iou_score(outputs, masks)
    
    return total_loss / len(test_loader), total_iou / len(test_loader)

torch.cuda.empty_cache() 

import segmentation_models_pytorch as smp

# early_stopping.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint2.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss.cpu()):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            # return self.counter

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Iou SCORE INCREASED ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        return self

# class Network(nn.Module):
#     def __init__(self,num_classes=2):
#         super().__init__()
#         self.first_conv = nn.Conv2d(3, 1, (1,1), 1, 1) # you could use e.g. a 1x1 kernel
        
#         self.model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=1.0)
#         ##self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
#     def forward(self, x):
#         #out = self.first_conv(x)
#         x = x.float()
#         out = self.model(x)
#         return out


# 메인 함수
def main():
    image_dir = "./dataset/BreastCancer/images"
    mask_dir = "./dataset/BreastCancer/mask"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(image_dir, mask_dir)
    
    model = UNet(num_classes=1,in_channels=3)
    model.to(device)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = smp.losses.DiceLoss(mode='binary')
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # EarlyStopping 객체 생성
    early_stopping = EarlyStopping(patience=30, verbose=True)
    num_epochs = 300
    for epoch in range(num_epochs):
        # 훈련
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # EarlyStopping을 적용해 성능 개선 여부 체크
        early_stopping(val_iou, model)

        # 스케줄러 스텝 업데이트
        # scheduler.step()
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")
            
        # EarlyStopping이 활성화되면 학습 종료
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 학습이 완료되었을 때 저장된 모델 로드
    model.load_state_dict(torch.load('checkpoint2.pt'))

    # 테스트 세트에서 평가
    test_loss, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")
    
    # 테스트 세트에서 첫 번째 이미지에 대한 예측 결과 시각화
    model.eval()
    test_images, test_masks = next(iter(test_loader))
    with torch.no_grad():
        pred_mask = model(test_images[0:1].to(device))
        # 출력값 범위 확인
        print("Raw prediction range:", torch.min(pred_mask).item(), torch.max(pred_mask).item())
        
        # sigmoid 후 범위 확인
        print("After sigmoid range:", torch.min(pred_mask).item(), torch.max(pred_mask).item())
        
        pred_mask = (pred_mask > 0.5).float()
        # 이진화 후 unique 값 확인
        print("After thresholding unique values:", torch.unique(pred_mask))
        
        pred_mask = pred_mask.cpu().squeeze()


    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(test_images[0].permute(1, 2, 0))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(132)
    # Ground Truth도 0-255로 정규화
    plt.imshow(test_masks[0].squeeze()*255, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(133)
    # 예측 마스크를 0-255로 정규화
    plt.imshow(pred_mask*255 , cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("prediction_testimage_unet++_1.png")  # 저장을 먼저 하고
    plt.show()  # 그 다음 화면에 표시
if __name__ == "__main__":
    main()
