import torch
import torch.nn as nn
from trlif import trLIFNode
from spikingjelly.activation_based import layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_tau=2.0, init_thr=2.0):
        super(BasicBlock, self).__init__()
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.sn1 = trLIFNode(init_thr=init_thr, init_tau=init_tau, step_mode='m', v_reset=None, detach_reset=False, decay_input=True)
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.sn2 = trLIFNode(init_thr=init_thr, init_tau=init_tau, step_mode='m', v_reset=None, detach_reset=False, decay_input=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = layer.SeqToANNContainer(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.sn2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, T=4, init_tau=2.0, init_thr=2.0, zero_init=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.T = T
        self.init_tau = init_tau
        self.init_thr = init_thr

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = trLIFNode(init_thr=init_thr, init_tau=init_tau, step_mode='m', v_reset=None, detach_reset=False, decay_input=True)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.avg_pool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = layer.SeqToANNContainer(nn.Linear(512*block.expansion, num_classes))
        
        if zero_init:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, init_thr=self.init_thr, init_tau=self.init_tau))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return torch.transpose(x, 0, 1)


def ResNet19(**kwargs):
    return ResNet(BasicBlock, [3, 3, 2], **kwargs)