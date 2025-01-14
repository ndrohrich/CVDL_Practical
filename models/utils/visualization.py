import torch
from torchvision.transforms import functional as TF
import numpy as np

def visualize_gradients(model, img, label, cfg):
    # Get blur size from cfg
    blur_size = cfg.blur_size
import cv2
import numpy as np
import torch


def visualize_gradients(model, img, label, cfg):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    #get blur_size from cfg
    blur_size = cfg.blur_size

    # Ensure the image requires gradient
    img.requires_grad = True

    # Ensure the label, img, and model are on the same device
    img = img.to(model.parameters().__next__().device)
    label = label.to(model.parameters().__next__().device)

    # Zero the model gradients
    model.zero_grad()

    # Forward pass
    if cfg.model == "fcn" or cfg.model == "ACN":
        features, output = model(img)
    else:
        output = model(img)

    # Compute loss
    loss = torch.nn.CrossEntropyLoss()(output, label)

    # Backward pass
    loss.backward()

    # Get the gradients
    grad = img.grad
    grad = torch.abs(grad)


    # process batch
    if len(grad.shape) == 3:
        grad = grad.unsqueeze(0)
        img = img.unsqueeze(0)
    grad_arr=[]
    img_arr=[]
    for i in range(grad.shape[0]):
        grad_arr.append(grad[i])
        img_arr.append(img[i])
    

    def process_gradients(grad, img, blur_size=11):
        # Smooth the grad
        grad = grad.squeeze().detach().cpu().numpy()
        grad = cv2.GaussianBlur(grad, (blur_size, blur_size), 0)
        grad = grad / grad.max()
        grad = grad * 255
        grad = grad.astype(np.uint8)
        grad = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        grad = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)

        # Prepare the original image
        img = img.squeeze().detach().cpu().numpy()
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        mapped_grad = cv2.addWeighted(img, 0.7, grad, 0.3, 0)

        return np.array(mapped_grad).transpose(2, 0, 1).reshape(1, 3, 64, 64)

    mapped_grad_arr = []
    for i in range(len(grad_arr)):
        mapped_grad_arr.append(process_gradients(grad_arr[i], img_arr[i], blur_size))
    mapped_grad_arr = np.concatenate(mapped_grad_arr, axis=0)

    return mapped_grad_arr