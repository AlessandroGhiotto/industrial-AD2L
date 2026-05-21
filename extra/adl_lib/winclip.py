import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import faiss

class WinCLIPPlus(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        self.patch_size = self.model.config.vision_config.patch_size
        self.hidden_size = self.model.config.vision_config.hidden_size
        self.projection_dim = self.model.config.projection_dim
        
        self.normal_text_embeds = None
        self.anomaly_text_embeds = None
        self.anomaly_descriptions = []
        
        self.normal_templates = [
            "a flawless photo of a {}",
            "a perfect {}",
            "a normal {}",
            "a good {}",
            "a photo of a {} without any defects"
        ]
        self.anomaly_template = "a {} with {}"
        
        # Memory Bank variables
        self.faiss_index = None
        
        # Semantic search variables
        self.labeled_cls_features = None
        self.labeled_defect_indices = None # indices matching anomaly_descriptions
        
    def setup_prompts(self, class_name: str, object_name: str, csv_path: str):
        df = pd.read_csv(csv_path)
        class_df = df[df['public_class'] == class_name]
        
        if len(class_df) == 0:
            print(f"Warning: No descriptions found for {class_name} in {csv_path}")
            self.anomaly_descriptions = ["anomalies or defects"]
        else:
            self.anomaly_descriptions = class_df['description'].dropna().unique().tolist()
            
        normal_prompts = [template.format(object_name) for template in self.normal_templates]
        anomaly_prompts = [self.anomaly_template.format(object_name, desc) for desc in self.anomaly_descriptions]
        
        # We also keep a generic anomaly prompt at the end
        self.anomaly_descriptions.append("anomalies or defects")
        anomaly_prompts.append(self.anomaly_template.format(object_name, "anomalies or defects"))
        
        with torch.no_grad():
            normal_inputs = self.processor(text=normal_prompts, return_tensors="pt", padding=True).to(self.device)
            anomaly_inputs = self.processor(text=anomaly_prompts, return_tensors="pt", padding=True).to(self.device)
            
            normal_text_outputs = self.model.text_model(**normal_inputs)
            normal_text_embeds = self.model.text_projection(normal_text_outputs[1])
            self.normal_text_embeds = F.normalize(normal_text_embeds, p=2, dim=-1)
            
            anomaly_text_outputs = self.model.text_model(**anomaly_inputs)
            anomaly_text_embeds = self.model.text_projection(anomaly_text_outputs[1])
            self.anomaly_text_embeds = F.normalize(anomaly_text_embeds, p=2, dim=-1)

    def extract_patch_features(self, pixel_values):
        vision_model = self.model.vision_model
        hidden_states = vision_model.embeddings(pixel_values)
        hidden_states = vision_model.pre_layrnorm(hidden_states)
        
        encoder_outputs = vision_model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=False
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = vision_model.post_layernorm(last_hidden_state)
        
        # CLS token is index 0, patches are the rest
        cls_feature = last_hidden_state[:, 0, :]
        patch_features = last_hidden_state[:, 1:, :] 
        
        # Project to CLIP space
        projected_cls = self.model.visual_projection(cls_feature)
        projected_patches = self.model.visual_projection(patch_features)
        
        projected_cls = F.normalize(projected_cls, p=2, dim=-1)
        # Note: Do not normalize patches here if we want to do multi-scale pooling later
        
        return projected_cls, projected_patches

    def fit_normal(self, loader, max_samples=25000):
        print("Extracting normal features for memory bank...")
        features = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Fit Normal"):
                images = batch["image"].to(self.device)
                _, patch_features = self.extract_patch_features(images) # [B, N, D]
                patch_features = F.normalize(patch_features, p=2, dim=-1) # Normalize here for FAISS
                features.append(patch_features.view(-1, self.projection_dim).cpu().numpy())
                
        features = np.concatenate(features, axis=0)
        
        # Simple random coreset to avoid exploding memory
        if len(features) > max_samples:
            idx = np.random.choice(len(features), max_samples, replace=False)
            features = features[idx]
            
        print(f"Building FAISS index with {len(features)} patches...")
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(self.projection_dim) # Inner product for cosine similarity (features are normalized)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            self.faiss_index = faiss.IndexFlatIP(self.projection_dim)
            
        self.faiss_index.add(features)
        
    def fit_labeled_anomalies(self, loader):
        print("Extracting labeled anomaly CLS features for semantic search...")
        cls_feats = []
        defect_indices = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Fit Labeled Anomalies"):
                images = batch["image"].to(self.device)
                defect_types = batch["defect_type"] # e.g. "anomaly_01"
                
                cls_features, _ = self.extract_patch_features(images)
                cls_feats.append(cls_features.cpu().numpy())
                
                for dt in defect_types:
                    # In CSV, public_anomaly is "anomaly_01", but the descriptions are just strings.
                    # We might need the exact description, but for now we just map defect type roughly.
                    # Actually, the user suggested linking the labeled example to the text description.
                    # For simplicity, we just use the index of the description. If we can't perfectly map "anomaly_01" to the exact string,
                    # we can map it to the generic anomaly.
                    # Wait, the dataset loader returns `defect_type` as a string.
                    # Let's just find the corresponding index in self.anomaly_descriptions if possible.
                    idx = len(self.anomaly_descriptions) - 1 # default generic
                    # We could read the CSV to map defect_type to description index.
                    # We will do this mapping in the evaluation script or here.
                    defect_indices.append(dt) # store the string, we'll map during forward
                    
        self.labeled_cls_features = np.concatenate(cls_feats, axis=0)
        self.labeled_defect_indices = defect_indices

        # Build mapping from defect_type to anomaly description index
        self.defect_to_desc_idx = {} # To be populated by evaluation script
            
    def forward(self, pixel_values, alpha=0.5):
        B, C, H, W = pixel_values.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        with torch.no_grad():
            cls_features, raw_patch_features = self.extract_patch_features(pixel_values)
            
            # Multi-scale patch features
            D_feat = self.projection_dim
            x = raw_patch_features.transpose(1, 2).view(B, D_feat, grid_h, grid_w)
            
            scales = []
            # Scale 1 (1x1)
            scales.append(F.normalize(raw_patch_features, p=2, dim=-1))
            
            # Scale 2 (2x2)
            x2 = F.avg_pool2d(F.pad(x, (0, 1, 0, 1)), kernel_size=2, stride=1)
            x2_flat = x2.view(B, D_feat, -1).transpose(1, 2)
            scales.append(F.normalize(x2_flat, p=2, dim=-1))
            
            # Scale 3 (3x3)
            x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            x3_flat = x3.view(B, D_feat, -1).transpose(1, 2)
            scales.append(F.normalize(x3_flat, p=2, dim=-1))
            
            anomaly_maps_text_ms = []
            anomaly_maps_memory_ms = []
            
            for patch_features in scales:
                # --- Text Score (Semantic Search) ---
                if self.labeled_cls_features is not None:
                    # Find nearest labeled anomaly
                    similarities = torch.matmul(cls_features, torch.from_numpy(self.labeled_cls_features).to(self.device).t()) # [B, M]
                    best_match_indices = similarities.argmax(dim=-1).cpu().numpy()
                    
                    am_text = []
                    for b in range(B):
                        best_match_idx = best_match_indices[b]
                        defect_type = self.labeled_defect_indices[best_match_idx]
                        desc_idx = self.defect_to_desc_idx.get(defect_type, len(self.anomaly_descriptions)-1)
                        specific_anomaly_embed = self.anomaly_text_embeds[desc_idx:desc_idx+1]
                        
                        p_feat = patch_features[b:b+1] # [1, N, D]
                        sim_norm = torch.matmul(p_feat, self.normal_text_embeds.t()).mean(dim=-1) # [1, N]
                        sim_anom = torch.matmul(p_feat, specific_anomaly_embed.t()).squeeze(-1) # [1, N]
                        
                        score = sim_anom - sim_norm
                        am_text.append(score)
                        
                    am_text = torch.cat(am_text, dim=0) # [B, N]
                else:
                    sim_normal = torch.matmul(patch_features, self.normal_text_embeds.t()).mean(dim=-1)
                    sim_anomaly, _ = torch.matmul(patch_features, self.anomaly_text_embeds.t()).max(dim=-1)
                    am_text = sim_anomaly - sim_normal
                
                anomaly_maps_text_ms.append(am_text)
                
                # --- Memory Score ---
                am_memory = torch.zeros_like(am_text)
                if self.faiss_index is not None:
                    p_feat_flat = patch_features.view(-1, self.projection_dim).cpu().numpy()
                    D, _ = self.faiss_index.search(p_feat_flat, 1)
                    D = 1.0 - D.reshape(B, -1) # 1 - Cosine Similarity
                    am_memory = torch.from_numpy(D).to(self.device)
                
                anomaly_maps_memory_ms.append(am_memory)
                
            # Average across scales
            anomaly_maps_text = torch.stack(anomaly_maps_text_ms, dim=0).mean(dim=0)
            anomaly_maps_memory = torch.stack(anomaly_maps_memory_ms, dim=0).mean(dim=0)
            
            # DO NOT min-max normalize per-image. Just use the raw similarities.
            # Fusion
            if self.faiss_index is not None and self.labeled_cls_features is not None:
                # Text score ranges from roughly -0.5 to 0.5. Memory score ranges from 0 to 1.
                # Adding them directly is mathematically sound since both are cosine similarities.
                anomaly_map = (1 - alpha) * anomaly_maps_text + alpha * anomaly_maps_memory
            elif self.faiss_index is not None:
                anomaly_map = anomaly_maps_memory
            else:
                anomaly_map = anomaly_maps_text

            anomaly_map = anomaly_map.view(B, 1, grid_h, grid_w)
            
            # Interpolate to original image size
            anomaly_map_resized = F.interpolate(
                anomaly_map, size=(H, W), mode='bilinear', align_corners=False
            )
            
            return anomaly_map_resized
