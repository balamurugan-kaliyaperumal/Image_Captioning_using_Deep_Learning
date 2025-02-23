import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Define Paths
IMAGE_FOLDER = "D:/project/dataset/images"
OUTPUT_FOLDER = "D:/project/dataset/feature_vectors"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ViT Parameters
PATCH_SIZE = 16
NUM_PATCHES = (224 // PATCH_SIZE) ** 2  # 14x14 = 196 patches
EMBED_DIM = 768  # Original ViT embedding dimension
REDUCED_DIM = 512  # Target dimension after projection

# Define the ViT Feature Extractor
class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        
        # Linear Projection from 768 -> 512
        self.proj = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, EMBED_DIM)
        self.reduce_dim = nn.Linear(EMBED_DIM, REDUCED_DIM)
        
        # Learnable Position Embedding + CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embedding = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, EMBED_DIM))
        
        # Transformer Encoder (Minimal for Feature Extraction)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        batch_size = x.shape[0]

        # Convert Image into 16x16 Patches
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, NUM_PATCHES, -1)  # Shape: [B, 196, 768]
        
        # Apply Linear Projection (768D)
        x = self.proj(x)

        # Add CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, 197, 768]

        # Add Position Embeddings
        x += self.pos_embedding

        # Transformer Encoder
        x = self.encoder(x)

        # Normalize Output
        x = self.norm(x)

        # Reduce Dimension to 512D
        x = self.reduce_dim(x[:, 0])  # Extract CLS token and reduce to 512D
        
        return x

# Initialize Model
vit_extractor = ViTFeatureExtractor()

# Function to Extract Features
def extract_features():
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(("png", "jpg", "jpeg")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            
            # Load and Transform Image
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            # Extract Features
            with torch.no_grad():
                feature_vector = vit_extractor(img_tensor)

            # Save Feature Vector (Shape: 512D)
            feature_vector_path = os.path.join(OUTPUT_FOLDER, f"{filename}_features.npy")
            np.save(feature_vector_path, feature_vector.numpy())
            
            print(f"Feature vector saved: {feature_vector_path}")

# Run Feature Extraction
extract_features()