import sys
sys.path.append('.')
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from adl_lib.config import PATH, CLASS_NAME, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED, seed_everything
from adl_lib.data import ADLTrainAnomalyLabeledDataset, ADLTrainGoodDataset, tensor_to_numpy_image, tensor_to_numpy_mask
from adl_lib.utils import summarize_metrics, postprocess_anomaly_map, save_results_to_pdf, calibrate_threshold_from_labeled
from adl_lib.winclip import WinCLIPPlus

def get_object_name(csv_path, class_name):
    try:
        df = pd.read_csv(csv_path)
        class_df = df[df['public_class'] == class_name]
        if len(class_df) > 0:
            return class_df.iloc[0]['object_name']
    except Exception as e:
        pass
    return "object"

def build_defect_mapping(winclip, class_name, csv_path):
    # Maps defect_type (e.g. "anomaly_01") to the index in winclip.anomaly_descriptions
    df = pd.read_csv(csv_path)
    class_df = df[df['public_class'] == class_name]
    
    mapping = {}
    for _, row in class_df.iterrows():
        defect = row['public_anomaly']
        desc = row['description']
        if desc in winclip.anomaly_descriptions:
            idx = winclip.anomaly_descriptions.index(desc)
            mapping[defect] = idx
    
    winclip.defect_to_desc_idx = mapping

def main():
    seed_everything(SEED)
    classes = ["class_01", "class_02", "class_03", "class_04", "class_05", "class_06", "class_07", "class_08"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading WinCLIP+ Model: {model_name} on {device}")
    winclip = WinCLIPPlus(model_name=model_name, device=device)
    
    csv_path = "./dataset/adl-2025-2026-anomaly-detection/anomaly_descriptions.csv"
    PATH_PRE = "./dataset/adl-2025-2026-anomaly-detection" # Use the original unmasked images for CLIP
    
    all_class_results = []
    
    for class_name in classes:
        print(f"\n" + "="*40)
        print(f"Results for {class_name}")
        print("="*40)
        
        object_name = get_object_name(csv_path, class_name)
        print(f"Object: {object_name}")
        
        try:
            winclip.setup_prompts(class_name, object_name, csv_path)
            build_defect_mapping(winclip, class_name, csv_path)
            
            # Memory Bank: Fit Normal
            try:
                good_ds = ADLTrainGoodDataset(PATH_PRE, class_name, image_size=IMAGE_SIZE)
                good_loader = DataLoader(good_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                winclip.fit_normal(good_loader)
            except Exception as e:
                print(f"Skipping FAISS memory bank due to: {e}")
                winclip.faiss_index = None
                
            # Few-Shot Semantic: Fit Labeled Anomalies
            anom_train_ds = ADLTrainAnomalyLabeledDataset(PATH_PRE, class_name, image_size=IMAGE_SIZE)
            anom_loader = DataLoader(anom_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            winclip.fit_labeled_anomalies(anom_loader)
            
            # To evaluate zero-shot fairly without training on test, we just evaluate on the labeled anomalies for quick check
            # Wait, evaluating on labeled anomaly set to see if it improved:
            # Let's load the same anom dataset for evaluation
            val_loader = DataLoader(anom_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            
            results = []
            
            for batch in tqdm(val_loader, desc=f"Evaluating WinCLIP+ for {class_name}"):
                images = batch["image"].to(device)
                
                # Use an alpha weighting (0.5 means equal weight to text and memory score)
                anomaly_maps = winclip(images, alpha=0.5) 
                anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy() # [B, H, W]
                
                for i in range(len(anomaly_maps)):
                    am = postprocess_anomaly_map(anomaly_maps[i], sigma=4.0, background_percentile=10.0)
                    score = float(am.max())
                    
                    results.append({
                        "anomaly_map": am,
                        "score": score,
                        "mask": tensor_to_numpy_mask(batch["mask"][i]),
                        "label": int(batch["label"][i].item()),
                        "image": tensor_to_numpy_image(images[i]),
                        "path": batch["path"][i]
                    })
            
            metrics = summarize_metrics(results)
            print(f"Pixel AP: {metrics['pixel_ap']:.4f}")
            print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")
            
            # Use threshold from metrics for visualization if available
            thr = metrics.get('pixel_threshold', 0.5)
            if np.isnan(thr): thr = 0.5
            
            all_class_results.append((class_name, results, thr))
            
        except Exception as e:
            print(f"Error processing {class_name}: {e}")
            
    print("\nGenerating PDF report...")
    save_results_to_pdf(all_class_results, "artifacts/winclip_predictions.pdf")
    print("Saved to artifacts/winclip_predictions.pdf")

if __name__ == "__main__":
    main()
