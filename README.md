# Image_Captioning_using_Deep_Learning


#Required Libraries
pip install Flask torch numpy Pillow werkzeug


1. Image Upload
2. storing the Uploaded image in uploads folder
3. convert to jpg format and store it in the upload folder
4. Resizing the image (224 x 224 pixels) and the resized image is stored in resized_images folder
5. Normalizing the image and the normalized image is stored in normalized_images folder
6. Extracted the Patches (196 patches) and the each patches are stored in patches folder
7. Flattened the Patches (2D shape to 1D vector)(768 tokens for each patch) and the resulted text file stored in the text_output folder
8. Apply Linear Projection (768 -> 512 tokens) and the output file stored in the text_output folder
9. Patch + Position embedding has been successfully completed and the output text file stored in the text_output folder
10. Feature vector extraction using ViT encoder (MHSA, Add Norm, FFN, Add Norm)
    run the app.py file to extract the feature vectors and the output is (196,512)
11. extract the feature vector for entire dataset and merged into a single file