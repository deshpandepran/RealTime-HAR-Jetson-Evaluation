from torch.utils.data import DataLoader
from data.activitynet import ActivityNetSubsetDataset
from data.fifty_salads import FiftySaladsDataset

# ActivityNet training loader (Distillation / IMFE pretraining)
activitynet_train = ActivityNetSubsetDataset(data_root="./data/activitynet", split="train")
train_loader = DataLoader(activitynet_train, batch_size=16, shuffle=True, num_workers=4)

# 50 Salads 5-fold cross-validation loader (Fold 1, Evaluation at 30 FPS)
salads_val = FiftySaladsDataset(data_root="./data/fifty_salads", fold=1, split="val", target_fps=30)
val_loader = DataLoader(salads_val, batch_size=1, shuffle=False, num_workers=2)