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
# import segmentation_models_pytorch as smp
# from timm.models.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
# import albumentations as album
from torchsummary import summary


'''
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path(U-net Encoder part)

        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        # kernel_size=3, stride=1, padding=1, bias=True --> 설정한 함수에 pre-defined

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path(U-net Decoder part)

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        # 2*256 --> skip connection 으로 연결되는 볼륨 추가
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        # 2*256 --> skip connection 으로 연결되는 볼륨 추가
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        # 2*256 --> skip connection 으로 연결되는 볼륨 추가
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        # n 개 클래스에 대한 output 만들어주기 위해 conv 1*1


    ## U-net layer 연결

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        # x = input_image

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

'''

# U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.down1 = conv_block(in_channels, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        
        self.bridge = conv_block(512, 1024)

        # upsample 부분에서 skip connection과 결합할 때 입력 채널을 고려하여 수정
        self.up4 = conv_block(1024 + 512, 512)  # 업샘플링된 1024 + 스킵 512
        self.up3 = conv_block(512 + 256, 256)  # 업샘플링된 512 + 스킵 256
        self.up2 = conv_block(256 + 128, 128)  # 업샘플링된 256 + 스킵 128
        self.up1 = conv_block(128 + 64, 64)    # 업샘플링된 128 + 스킵 64

        
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        d1 = self.down1(x)
        x = self.maxpool(d1)
        
        d2 = self.down2(x)
        x = self.maxpool(d2)
        
        d3 = self.down3(x)
        x = self.maxpool(d3)
        
        d4 = self.down4(x)
        x = self.maxpool(d4)
        
        x = self.bridge(x)
        
        x = self.upsample(x)

        x = torch.cat([x, d4], dim=1)
        x = self.up4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, d3], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x)
        
        return torch.sigmoid(self.final(x))

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
    # input_image = Image.open(filename)
    # m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 이미지를 고정 크기로 리사이즈
        transforms.ToTensor(),
        # transforms.Normalize(mean=m, std=s),
    ])
    
    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)


    # 빈 데이터셋 체크
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your image and mask directories.")
    
    # train, validation, test 데이터로 나눔
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)  # 시드 고정
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 시드 고정 (train 60%, val 20%, test 20%)
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


# IoU 계산 함수
def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)
from torch.cuda.amp import autocast, GradScaler

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice 

# 훈련 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    accumulation_steps = 4  # gradient accumulation 단계
    scaler = GradScaler()

    for images, masks in tqdm(iter(train_loader)):
        images, masks = images.to(device, torch.float32), masks.to(device, torch.float32)
        optimizer.zero_grad()
        outputs = model(images)
        # print(images.shape)
        # print(masks.shape)
        # print(outputs.shape)
        # masks = masks[:,:2,:,:]
        # masks = torch.cat([masks, masks], dim=1)

        # print(masks)
        # loss = criterion(outputs, masks)

        # loss = loss / accumulation_steps

        # loss.backward()
        # # accumulation steps마다 optimizer step
        # if (i + 1) % accumulation_steps == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        # total_loss += loss.item() * accumulation_steps

        optimizer.zero_grad()
        
        # mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()




    return total_loss / len(train_loader)

import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# 새로운 validation 평가 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in tqdm(iter(val_loader)):
            images, masks = images.to(device), masks.to(device)
            # masks = torch.cat([masks, masks], dim=1)
            dice = Dice(average='micro')
            dice(outputs, masks)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += iou_score(outputs, masks)
            # total_dice +=dice(outputs, masks)
            
            

    return model, total_loss / len(val_loader), total_iou / len(val_loader), total_dice / len(val_loader)

# 평가 함수
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            # masks = torch.cat([masks, masks], dim=1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += iou_score(outputs, masks)
    
    return total_loss / len(test_loader), total_iou / len(test_loader)
# early_stopping.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint1.pt', trace_func=print):
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

torch.cuda.empty_cache() 

import segmentation_models_pytorch as smp


from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchmetrics.classification import Dice


# 메인 함수
def main():
    image_dir = "./dataset/BreastCancer/images"
    mask_dir = "./dataset/BreastCancer/mask"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(image_dir, mask_dir)
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=1.0)
    model = UNet()
    # summary(model, input_size = (3,256,256))
    # first_conv = nn.Conv2d(3, 1, kernel_size, stride, padding) # you could use e.g. a 1x1 kernel
    # model = pretrained_model()

    # x = # load data, should have the shape [batch_size, 3, height, width]
    # out = first_conv(x)
    # out = pretrained_model(out)
    # pretrained encoder를 사용하는 UNet 모델 생성
    '''
    model = smp.Unet(
        encoder_name="resnet34",        # resnet34, resnet50 등 선택 가능
        encoder_weights="imagenet",     # pretrained weights
        in_channels=3,                  # 입력 이미지 채널 수
        classes=1,                      # 분할할 클래스 수
    )
    '''
    model.to(device)


    '''
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )
    '''
    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = smp.losses.DiceLoss(mode='binary')
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    

    # EarlyStopping 객체 생성
    early_stopping = EarlyStopping(patience=40, verbose=True)

    num_epochs = 300
    for epoch in range(num_epochs):
        # 훈련
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # 검증
        model, val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
        
        # EarlyStopping을 적용해 성능 개선 여부 체크
        early_stopping(val_iou, model)

        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}, Validation Dice: {val_dice:.4f}")

        
        # EarlyStopping이 활성화되면 학습 종료
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 스케줄러 스텝 업데이트
        # scheduler.step()


    # 학습이 완료되었을 때 저장된 모델 로드
    model.load_state_dict(torch.load('checkpoint1.pt'))
    
    # 테스트 세트에서 평가
    test_loss, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")
    
    # 테스트 세트에서 첫 번째 이미지에 대한 예측 결과 시각화
    model.eval()
    test_images, test_masks = next(iter(test_loader))
    with torch.no_grad():
        pred_mask = model(test_images[0:1].to(device))
        # sigmoid 적용하고 0-1 사이 값으로 변환
        pred_mask = (pred_mask > 0.5).float()  # 임계값 0.5로 이진화
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
    plt.imshow(pred_mask *255, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("prediction_testimage_unet1.png")  # 저장을 먼저 하고
    plt.show()  # 그 다음 화면에 표시
if __name__ == "__main__":
    main()
