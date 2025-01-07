import torch
import torch.nn as nn

import torch
import torch.nn as nn
class AffinityLoss(nn.Module):
    def __init__(self, feature_dim, num_classes):
        """
        Initializes the AffinityLoss module.

        Args:
            feature_dim (int): The dimensionality of the feature vectors.
            num_classes (int): The number of classes in the classification task.
        """
        super(AffinityLoss, self).__init__()
        self.num_classes = num_classes


    def forward(self, features, labels):
        """
        Computes the affinity loss.

        Args:
            features (torch.Tensor): Batch of feature vectors of shape (N, D), where N is the batch size and D is the feature dimension.
            labels (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The computed affinity loss.
        """
        # Get the batch size and feature dimension
        batch_size, feature_dim = features.size()
        
        self.centers=torch.zeros(self.num_classes,feature_dim,device=features.device)
        _labels = torch.argmax(labels,dim=1)
        # print(f"center device: {self.centers.device}")
        # print(f"features device: {features.device}")

        #recalculate the centers
        for i in range(self.num_classes):
            # transfer labeln from one hot back
            mask=_labels==i        
        
            # print(f"class {i}")
            # print(f"mask: {mask.shape,mask}")
            if mask.sum()==0:
                continue
            self.centers[i,:] = features[mask].mean(dim=0)
        
        self.centers=self.centers+1e-6
        # print(f"centers: {self.centers.shape}")
        # Get the centers corresponding to the ground truth labels
        centers_batch = self.centers[_labels,:]
        # print(f"centers_batch: {centers_batch.shape}")

        # Compute intra-class distance (numerator)
        intra_class_dist = torch.sum((features - centers_batch) ** 2, dim=1)

        # Compute standard deviation of class centers (denominator)
        center_mean = self.centers.mean(dim=0, keepdim=True)
        inter_class_var = torch.sum((self.centers - center_mean) ** 2) / self.num_classes
        
        # print(f"centers: {self.centers.shape,self.centers}")
        # print(f"centers_mean: {center_mean.shape,center_mean}")
        # print(f"center_diff: {torch.sum((self.centers - center_mean) ** 2)+1e-6 }")
        # print(f"num_cl: {self.num_classes}")
        # print(f"sum dist: {torch.sum(intra_class_dist)}")
        # print(f"interclass_var: {(inter_class_var + 1e-6)}")
        # Compute the affinity loss
        loss = torch.sum(intra_class_dist) / (inter_class_var + 1e-6)  # Add epsilon for numerical stability

        return loss


class combined_loss(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha=1.0, beta=1.0):
        """
        Initializes the combined loss module.

        Args:
            feature_dim (int): The dimensionality of the feature vectors.
            num_classes (int): The number of classes in the classification task.
            alpha (float): The weight for the cross-entropy loss.
            beta (float): The weight for the affinity loss.
        """
        super(combined_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()
        self.affinity_loss = AffinityLoss(feature_dim, num_classes)

    def forward(self, features,outputs,labels):
        """
        Computes the combined loss.

        Args:
            features (torch.Tensor): Batch of feature vectors of shape (N, D), where N is the batch size and D is the feature dimension.
            labels (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The computed combined loss.
        """
        #print(features,outputs)
        
        # Compute the cross-entropy loss
        cross_entropy_loss = self.cross_entropy(outputs, labels)
        #print(f"ce_loss: {cross_entropy_loss}")

        # Compute the affinity loss
        affinity_loss = self.affinity_loss(features, labels)
        #print(f"a_loss: {affinity_loss}")

        # Compute the combined loss
        loss = self.alpha * cross_entropy_loss + self.beta * affinity_loss
        
    

        return loss