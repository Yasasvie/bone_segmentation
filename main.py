from args import get_args
import os
import pandas as pd
import torch
from dataset import knee_dataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model

def main():
    args = get_args()

    # 1. reading csv files
    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # 2. preparing dataset
    train_dataset = knee_dataset(train_set)
    val_dataset = knee_dataset(val_set)

    # 3. initializing the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # shuffle true for train_set to avoid overfitting
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)     # shuffle false for validation as it is not needed

    # 4. initializing the model
    model = UNetLext(input_channels=1,
                     output_channels=1,
                     pretrained=False,
                     path_pretrained='',
                     restore_weights=False,
                     path_weights='')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(model, train_loader, val_loader, device)

    # Save the trained model
    os.makedirs('session', exist_ok=True)
    model_save_path = 'session/best_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    main()