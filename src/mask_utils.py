import torch

class Mask(torch.nn.Module):
    def __init__(self, im_size, rad):
        super(Mask, self).__init__()

        mask = (torch.linspace(-1, 1, im_size)[None] ** 2 + torch.linspace(-1, 1, im_size)[:, None] ** 2) < rad
        self.register_buffer('mask', mask)

    def forward(self, x):
        return x * self.mask
