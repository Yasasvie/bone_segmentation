import os
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

script_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.abspath(os.path.join(script_dir, ".."))

xrays_dir = os.path.join(base_dir, "xrays")
masks_dir = os.path.join(base_dir, "masks")
csv_dir = os.path.join(base_dir, "csvs")

os.makedirs(csv_dir, exist_ok=True)

if not os.path.exists(xrays_dir):
    raise FileNotFoundError(f"Folder not found: {xrays_dir}")
if not os.path.exists(masks_dir):
    raise FileNotFoundError(f"Folder not found: {masks_dir}")

print("Folders found!")

print("Creating dataset.csv ...")

xrays = sorted([
    f for f in os.listdir(xrays_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

data = []
for xray in xrays:
    xray_path = f"data/xrays/{xray}"
    mask_path = f"data/masks/{xray}"

    if os.path.exists(os.path.join(masks_dir, xray)):
        data.append({"xrays": xray_path, "masks": mask_path})
    else:
        data.append({"xrays": xray_path, "masks": None})

dataset_csv = os.path.join(csv_dir, "dataset.csv")
df = pd.DataFrame(data)
df.to_csv(dataset_csv, index=False)

print(f"dataset.csv created with {len(df)} rows")

print("Splitting into train / val / test ...")

labeled = df[df["masks"].notna()].copy()

unlabeled = df[df["masks"].isna()].copy()

labeled = labeled.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.8 * len(labeled))
train_df = labeled.iloc[:train_size]
val_df = labeled.iloc[train_size:]
test_df = unlabeled

train_df.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

print(f"train.csv ({len(train_df)})")
print(f"val.csv ({len(val_df)})")
print(f"test.csv ({len(test_df)})")

counts = {
    "train": len(train_df),
    "validation": len(val_df),
    "test": len(test_df)
}

plt.bar(counts.keys(), counts.values(), color=["skyblue", "orange", "lightgreen"])
plt.title("Number of Images in Each CSV")
plt.ylabel("Count")
plt.show()

zip_name = "csv_files_group1.zip"
zip_path = os.path.join(base_dir, zip_name)

with zipfile.ZipFile(zip_path, "w") as zipf:
    for csv_file in ["dataset.csv", "train.csv", "val.csv", "test.csv"]:
        file_path = os.path.join(csv_dir, csv_file)
        zipf.write(file_path, arcname=csv_file)

print(f"ZIP file created at: {zip_path}")
