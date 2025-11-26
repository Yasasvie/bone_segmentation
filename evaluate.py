import os
import cv2
import torch
import numpy as np
from model import UNetLext
from args import get_args

args = get_args()
CSV_DIR = args.csv_dir
PRED_DIR = 'predictions/'
MODEL_PATH = 'session/best_model.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
test_csv = os.path.join(CSV_DIR, 'test.csv')
test_df = pd.read_csv(test_csv)

test_images = test_df['xrays'].tolist()

model = UNetLext(input_channels=1, output_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

def preprocess_image(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    xray = xray.reshape(1, 1, *xray.shape)
    return torch.tensor(xray, device=DEVICE)

def postprocess_mask(logits):
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).cpu().numpy()[0,0] * 255
    return mask.astype(np.uint8)

for img_path in test_images:
    xray_tensor = preprocess_image(img_path)
    with torch.no_grad():
        pred_logits = model(xray_tensor)
        mask = postprocess_mask(pred_logits)

    xray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    mask_resized = cv2.resize(mask, (xray.shape[1], xray.shape[0]))

    overlay = cv2.addWeighted(xray, 0.7, mask_resized, 0.3, 0)

    combined = np.hstack([xray, overlay])

    cv2.imshow("X-ray | Predicted Mask Overlay", combined)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()


