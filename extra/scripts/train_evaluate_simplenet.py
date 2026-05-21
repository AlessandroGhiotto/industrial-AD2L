import sys
sys.path.append('.')
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from adl_lib.config import PATH, CLASS_NAME, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED, seed_everything
from adl_lib.data import ADLTrainAnomalyLabeledDataset, ADLTrainGoodDataset
from adl_lib.utils import summarize_metrics, postprocess_anomaly_map, save_results_to_pdf
from adl_lib.simplenet import SimpleNetAD
import os

def train_simplenet(model, train_loader, epochs=20, lr=0.0002, device="cuda"):
    model.to(device)
    # We only train the discriminator in the simplest version
    # The adapter can just be a random projection if not pre-trained
    # Actually, random projection adapter works exceptionally well!
    optimizer = Adam(model.discriminator.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            
            optimizer.zero_grad()
            logits, labels = model(images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")

def evaluate_simplenet(model, val_loader, device="cuda"):
    model.eval()
    model.to(device)
    
    results = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()
            
            anomaly_maps = model(images)
            anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy() # [B, H, W]
            
            for i in range(len(anomaly_maps)):
                # SimpleNet output is logits. We can convert to prob or just use logits.
                # Usually applying a small gaussian smoothing helps locally.
                am = postprocess_anomaly_map(anomaly_maps[i], sigma=4.0)
                score = float(am.max())
                
                results.append({
                    "anomaly_map": am,
                    "score": score,
                    "mask": masks[i],
                    "label": int(labels[i]),
                    "image": images[i].cpu(),
                    "path": batch["path"][i]
                })
                
    return results

def main():
    seed_everything(SEED)
    class_name = "class_08"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    PATH_PRE = "./dataset/adl-2025-2026-anomaly-detection_birefnet"
    
    print(f"Loading SimpleNet Model on {device}")
    model = SimpleNetAD(backbone_name="wide_resnet50_2", adapter_dim=512, noise_std=0.5)
    
    good_ds = ADLTrainGoodDataset(PATH_PRE, class_name, image_size=IMAGE_SIZE)
    good_loader = DataLoader(good_ds, batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print("--- Training Phase ---")
    train_simplenet(model, good_loader, epochs=15, lr=0.0002, device=device)
    
    print("\n--- Evaluation Phase ---")
    anom_train_ds = ADLTrainAnomalyLabeledDataset(PATH_PRE, class_name, image_size=IMAGE_SIZE)
    val_loader = DataLoader(anom_train_ds, batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    results = evaluate_simplenet(model, val_loader, device=device)
    
    metrics = summarize_metrics(results)
    print(f"\nPixel AP: {metrics['pixel_ap']:.4f}")
    print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    
    os.makedirs("artifacts", exist_ok=True)
    thr = metrics.get('pixel_threshold', 0.5)
    if np.isnan(thr): thr = 0.5
    
    all_class_results = [(class_name, results, thr)]
    save_results_to_pdf(all_class_results, "artifacts/simplenet_predictions.pdf")
    print("Saved PDF to artifacts/simplenet_predictions.pdf")

if __name__ == "__main__":
    main()
