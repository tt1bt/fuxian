import torch
import torch.nn as nn
import torch.nn.functional as F


class CentripetalLoss(nn.Module):
    """
    Centripetal loss (CIAH):
    - Use fixed class centers (predefined from untrained/pretrained hashes).
    - Encourage hash codes to be close to their class center via cosine distance.
    """

    def __init__(self, num_classes, hash_bits, gamma=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.hash_bits = hash_bits
        self.gamma = gamma
        self.register_buffer("centers", torch.zeros(num_classes, hash_bits))

    @torch.no_grad()
    def set_centers(self, centers):
        if centers.shape != self.centers.shape:
            raise ValueError(
                f"centers shape {centers.shape} != {self.centers.shape}"
            )
        self.centers.copy_(centers)

    def forward(self, hash_code, labels):
        h = F.normalize(hash_code, dim=1)
        c = F.normalize(self.centers, dim=1)
        cos_sim = h @ c.t()
        logits = self.gamma * cos_sim
        loss = F.cross_entropy(logits, labels)
        return loss
