
# Script to preprocess dataset for image captioning: loads annotations, ViT features, tokenizes captions, and creates a batched TensorFlow dataset.
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_dataset():
    # Print TensorFlow version for compatibility check
    print("TensorFlow Version:", tf.__version__)

    # Load annotations
    with open(r"D:\Project\dataset\annotations_english.json", "r") as file:
        captions_data = json.load(file)

    # Load ViT features
    vit_features = np.load(r"D:\Image_Captioning_using_Deep_Learning\vit_features_merged.npy")
    print("✅ ViT Features Shape (raw):", vit_features.shape)

    # Extract unique image filenames
    unique_image_filenames = list(captions_data.keys())
    num_unique_images = len(unique_image_filenames)
    num_features = vit_features.shape[0]
    print(f"Number of unique images in annotations: {num_unique_images}")
    print(f"Number of features in vit_features_merged.npy: {num_features}")

    # Check for mismatch and trim if necessary
    if num_unique_images != num_features:
        print(f"⚠️ Mismatch detected: {num_unique_images} images vs {num_features} features")
        if num_features > num_unique_images:
            print(f"Trimming {num_features - num_unique_images} excess features from vit_features_merged.npy")
            vit_features = vit_features[:num_unique_images]
        else:
            raise ValueError("Fewer features than images; cannot proceed without complete feature set")

    # Align features with unique image filenames
    image_to_feature = {filename: vit_features[i] for i, filename in enumerate(unique_image_filenames)}

    # Create caption list and corresponding image features
    image_filenames = [img for img, details in captions_data.items() for _ in details["comments"]]
    caption_texts = [f"<start> {comment} <end>" for details in captions_data.values() for comment in details["comments"]]
    vit_features_expanded = np.array([image_to_feature[filename] for filename in image_filenames], dtype=np.float32)

    print("✅ Number of Captions:", len(caption_texts))
    print("✅ ViT Features Expanded Shape:", vit_features_expanded.shape)

    # Debug feature expansion
    print("Debugging vit_features_expanded:")
    for i in range(min(10, len(image_filenames))):
        print(f"Index {i}: Mean = {np.mean(vit_features_expanded[i])}, Filename = {image_filenames[i]}")

    # Tokenize captions
    vocab_size = 15000  # Match your model's vocab_size
    tokenizer = Tokenizer(num_words=vocab_size - 2, oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(caption_texts)
    tokenizer.word_index = {word: idx + 2 for word, idx in tokenizer.word_index.items()}
    tokenizer.word_index["<start>"] = 1
    tokenizer.word_index["<end>"] = 2
    tokenizer.index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    tokenizer.word_counts["<start>"] = len(caption_texts)
    tokenizer.word_counts["<end>"] = len(caption_texts)

    sequences = tokenizer.texts_to_sequences(caption_texts)
    max_length = 80
    padded_captions = pad_sequences(sequences, maxlen=max_length, padding="post")

    print("✅ Padded Captions Shape:", padded_captions.shape)
    print("Max token value in padded_captions:", np.max(padded_captions))
    print("Min token value in padded_captions:", np.min(padded_captions))

    # Verify special tokens
    print("Start token ID:", tokenizer.word_index["<start>"])
    print("End token ID:", tokenizer.word_index["<end>"])

    # Normalize ViT features
    vit_features_expanded = (vit_features_expanded - np.mean(vit_features_expanded, axis=0)) / np.std(vit_features_expanded, axis=0)

    # Prepare dataset with padding
    input_captions = padded_captions[:, :-1]  # [158817, 79]
    output_captions = padded_captions[:, 1:]  # [158817, 79]

    dataset = tf.data.Dataset.from_tensor_slices(((input_captions, vit_features_expanded), output_captions))
    dataset = dataset.shuffle(1000).padded_batch(
        batch_size=32,
        padded_shapes=(([79], [512]), [79]),
        padding_values=((0, 0.0), 0)
    ).prefetch(tf.data.AUTOTUNE)

    # Debug dataset shapes
    for i, (inputs, outputs) in enumerate(dataset.take(5)):
        decoder_inputs, encoder_outputs = inputs
        print(f"Batch {i} - decoder_inputs shape: {decoder_inputs.shape}")
        print(f"Batch {i} - encoder_outputs shape: {encoder_outputs.shape}")
        print(f"Batch {i} - outputs shape: {outputs.shape}")

    print("✅ Dataset prepared successfully!")

    # Save tokenizer
    with open(r"D:\Image_Captioning_using_Deep_Learning\tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())
    print("Tokenizer saved at D:\\Image_Captioning_using_Deep_Learning\\tokenizer.json")

    print("\nDebug: First 5 unique image filenames:", unique_image_filenames[:5])

    return dataset, tokenizer, image_filenames

if __name__ == "__main__":
    dataset, tokenizer, image_filenames = create_dataset()