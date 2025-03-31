from update import BasicUpdateBlock
from extractor import BasicEncoder

import torch
import torch.nn as nn


class DeepEraser(nn.Module):
    def __init__(self):
        super(DeepEraser, self).__init__()
        self.hidden_dim = 16
        self.context_dim = 8

        self.fnet = BasicEncoder(in_dim=4, output_dim=self.hidden_dim + self.context_dim, norm_fn="instance")
        self.update_block = BasicUpdateBlock(channels=3, hidden_dim=self.hidden_dim, input_dim=self.hidden_dim + self.context_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, mask, iters=64):
        image1 = image1.contiguous()
        image2 = torch.cat([image1, mask], dim=1)
        hdim = self.hidden_dim
        cdim = self.context_dim
        fmap1 = self.fnet(image2)

        net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        rec_image = image1
        for _ in range(iters):
            net, d_rec_image = self.update_block(net, inp, rec_image)
            rec_image = rec_image + d_rec_image * mask

        return rec_image
