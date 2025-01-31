import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAMHandler:
    def __init__(self, model, device='cuda'):
        """
        Initialize Grad-CAM handler.
        Args:
            model: The trained PyTorch Hybrid model.
            device: The device to use for computation.
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def compute_gradcam(self, input_tensor, target_class=None):
        """
        Compute Grad-CAM heatmap for the Hybrid model.
        Args:
            input_tensor: The input tensor of shape (1, C, H, W).
            target_class: The target class index for which Grad-CAM is computed. If None, the most likely class is used.
        Returns:
            heatmap_colored: The Grad-CAM heatmap in color (for overlay).
        """
        # Ensure input_tensor requires gradients
        input_tensor.requires_grad = True
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor.to(self.device))

        # Use the most likely class if target_class is not provided
        if target_class is None:
            target_class = torch.argmax(output)

        # Compute gradients
        class_score = output[:, target_class]
        class_score.backward()

        # Extract gradients and features
        gradients = self.model.gradients  
        features = self.model.features  

        # Compute Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)  
        cam = (weights * features).sum(dim=1).squeeze(0) 
        cam = F.relu(cam).detach().cpu().numpy()  

        # Normalize the heatmap
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  
        cam = np.uint8(255 * cam)  

        # Resize and apply colormap
        heatmap_resized = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        return heatmap_colored
