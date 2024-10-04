import   进口 time

import   进口 torch
import   进口 torch.nn as nn
from torch import   进口 optim
import   进口 timeit
from utils import   进口 get_loader
from model import   进口 VIT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
EPOCHS = 50
BATCH_SIZE = 16
TRAIN_DIR = './dataset/train.csv'
TEST_DIR = './dataset/test.csv'
SUBMISSION_DIR = './data/submission'

# Model Parameters   #模型参数
IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = (IMG_SIZE**2)*IN_CHANNELS
NUM_PATCHES = (IMG_SIZE//PATCH_SIZE)**2Num_patches = (img_size // patch_size)**
DROPOUT = 0.001   Dropout = 0.001   Dropout = 0.001

NUM_HEADS = 8   Num_heads = 8   Num_heads = 8
ACTIVATION = 'gelu'   激活= 'gelu'
NUM_ENCODERS = 768
NUM_CLASSES = 10   Num_classes = 10   Num_classes = 10

LEARNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BEATS = (0.9, 0.999)

train_dataloader, val_dataloader, test_dataloader = get_loader(TRAIN_DF_DIR, TEST_DF_DIR, SUBMISSION_DF_DIR, BATCH_SIZE)

model = VIT   NS 系列(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY, betas=ADAM_BEATS)

start_time = timeit.default_timer()

for   为 epoch in   在 range(EPOCHS):
    model.train   火车()
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for   为 idx, img_label in   在 enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["image"].float().to(DEVICE)
        label = img_label["label"].type   类型(torch.uint8).to(DEVICE)
        y_pred = model   模型(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion   标准(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for   为 idx, img_label in   在 enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(DEVICE)
            label = img_label["label"].type   类型(torch.uint8).to(DEVICE)
            y_pred = model   模型(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion   标准(y_pred, label)
            val_running_loss += loss.item()
    train_loss = val_running_loss / (idx + 1)

print   打印("-"*30)
print   打印(f"Train Loss Epoch {epoch + 1}: {train_loss:.4f}")
print   打印(f"Val Loss Epoch {epoch + 1}: {val_loss:.4f}")
print   打印(f"Train Accuracy Epoch {epoch + 1}: {sum(1 for   为 x ,y in   在 zip(train_preds, train_labels) if x==y) / len(train_labels):.4f}")
print   打印(f"Val Accuracy Epoch {epoch + 1}: {sum(1 for   为 x ,y in   在 zip(val_preds, val_labels) if x==y) / len(val_labels):.4f}")
print   打印("-"*30)

stop = timeit.default_timer()
print   打印(f"Training time: {stop - start}")

