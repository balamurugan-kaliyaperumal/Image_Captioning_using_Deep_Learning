import numpy as np
import os

# Path to the folder containing individual feature vector files
feature_vector_folder = "D:/project/dataset/feature_vectors" # CHANGE THIS

# Get all .npy files in the folder
feature_files = sorted([os.path.join(feature_vector_folder, f) for f in os.listdir(feature_vector_folder) if f.endswith(".npy")])

# Load and combine all feature vectors
all_features = [np.load(f) for f in feature_files]  # Each file contains (1, 512)

# Stack them into a single (N, 512) array
all_features = np.vstack(all_features)

# Save the merged features
np.save("vit_features_merged.npy", all_features)

print(f"Feature merging complete. Saved {all_features.shape[0]} feature vectors to vit_features.npy")
