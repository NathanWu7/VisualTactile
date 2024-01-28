import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import torch


class AdapterNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D=4):
        super(AdapterNetwork, self).__init__(D)

        channels = [32, 64, 128, 256]
        # conv
        conv_layers = []
        for l in range(len(channels)):
            if l == 0:
                conv_layers.append(ME.MinkowskiConvolution(
                    in_channels=in_feat,
                    out_channels=channels[l],
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    bias=False,
                    dimension=D),)
                # conv_layers.append(ME.MinkowskiConvolution(
                #     in_channels=channels[l],
                #     out_channels=channels[l],
                #     kernel_size=3,
                #     stride=1,
                #     dilation=1,
                #     bias=False,
                #     dimension=D),)
            else:
                conv_layers.append(ME.MinkowskiConvolution(
                    in_channels=channels[l-1],
                    out_channels=channels[l],
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    bias=False,
                    dimension=D),)      
                # conv_layers.append(ME.MinkowskiConvolution(
                #     in_channels=channels[l],
                #     out_channels=channels[l],
                #     kernel_size=3,
                #     stride=1,
                #     dilation=1,
                #     bias=False,
                #     dimension=D),)                 
            conv_layers.append(ME.MinkowskiBatchNorm(channels[l]))
            conv_layers.append(ME.MinkowskiELU())

        self.conv = nn.Sequential(*conv_layers)
        self.pooling = ME.MinkowskiGlobalPooling(ME.PoolingMode.GLOBAL_AVG_POOLING_KERNEL)
        self.linear = ME.MinkowskiLinear(256, out_feat)


    def forward(self, x):
        out = self.conv(x)
        #print('conv: ', out.coordinates.size(), out.features.size())
        out = self.pooling(out)
        #print('pooling: ', out.coordinates.size(), out.features.size())
        out = self.linear(out)
        #print('linear: ', out.coordinates.size(), out.features.size())

        return out

#TODO env encoder

if __name__ == '__main__':
    device = "cuda:0"


    pcs = torch.ones(2,4096,11).uniform_(0,1).to(device)
    labels = torch.ones(2,32).to(device)

    points = pcs[:,:,0:4]
    feats = pcs[:,:,4:]

    coords, feats = ME.utils.sparse_collate([point for point in points], [action for action in feats])
    input = ME.SparseTensor(feats, coordinates=coords, device=device)

    net = AdapterNetwork(in_feat=7, out_feat=32).to(device)
    print(net)
    output = net(input)
    print(output.size())