
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.transform import resize

# Adjust color mapping for Grad-CAM visualization
def apply_custom_colormap(cam_heatmap):
    # Normalize the heatmap between 0 and 1
    cam_heatmap_normalized = (cam_heatmap - np.min(cam_heatmap)) / (np.max(cam_heatmap) - np.min(cam_heatmap) + 1e-5)
    
    # Create the custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap', [(0, 'blue'), (0.25, 'green'), (0.5, 'yellow'), (0.75, 'orange'), (1, 'red')]
    )
    
    # Apply colormap
    cam_heatmap_colored = cmap(cam_heatmap_normalized)
    
    # Return RGB channels only (discard the alpha channel)
    return cam_heatmap_colored[..., :3]

# Choose the last convolutional layer you want to visualize
layer_to_visualize = model.vgg16[29]  # Last conv layer of VGG16

# Get a batch of images and labels from the test loader
data_iter = iter(test_loader)
images, labels = next(data_iter)  # Use next() to retrieve a batch
images, labels = images.to(device), labels.to(device)

# Instantiate the Grad-CAM algorithm
grad_cam = LayerGradCam(model, layer_to_visualize)

# Define class names (assuming 2 classes: Healthy and Patient)
class_names = ["Healthy", "Patient"]

  # Select a few random images from the test batch
  rand_indices = np.random.choice(len(images), size=10, replace=False)

  # Loop through random images and visualize Grad-CAM for each
  for i in rand_indices:
      img = images[i:i+1]  # Extract a single image
      label = labels[i]  # Get the true label for the image

      # Get the model's prediction for this image
      output = model(img)
      _, predicted = torch.max(output, 1)  # Get the predicted label

      # Apply Grad-CAM
      attribution = grad_cam.attribute(img, target=label.item())  # Compute Grad-CAM heatmap

      # Process the image for visualization
      img_np = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Convert to numpy (H, W, C)
      img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)  # Normalize for display

      # Process the Grad-CAM attribution map (2D heatmap)
      cam_heatmap = attribution.squeeze().detach().cpu().numpy()  # (H, W)

      # Resize the heatmap to match the image size
      cam_heatmap_resized = resize(cam_heatmap, (img_np.shape[0], img_np.shape[1]), anti_aliasing=True)  # Resize to original image size

      # Apply the custom colormap to the resized heatmap
      cam_heatmap_colored = apply_custom_colormap(cam_heatmap_resized)  # Apply custom colormap
      cam_heatmap_colored = (cam_heatmap_colored * 255).astype(np.uint8)  # Convert to uint8

      # Normalize the colored heatmap to ensure the higher values get more prominence
      cam_heatmap_colored = cam_heatmap_colored / 255.0
      cam_heatmap_colored = np.clip(cam_heatmap_colored, 0, 1)  # Ensure values are in range

      # Overlay the heatmap on the original image
      overlay = (0.5 * img_np + 0.5 * cam_heatmap_colored)  # Blend images with equal weights
      overlay = (overlay * 255).astype(np.uint8)  # Convert back to uint8

      # Plot the image and overlay
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.imshow(img_np)
      plt.title(f'True: {class_names[label]}, Predicted: {class_names[predicted]}')
      plt.axis('off')

      plt.subplot(1, 2, 2)
      plt.imshow(overlay)
      plt.title('Grad-CAM Overlay')
      plt.axis('off')

      plt.show()
