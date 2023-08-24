from torch import nn, flatten


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, dropout_p):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
    
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.conv1(input)
        input = self.dropout1(input)
        input = self.bn1(input)
        input = nn.ReLU()(self.bn1(self.dropout1(self.conv1(input))))
        input = nn.ReLU()(self.bn2(self.dropout2(self.conv2(input))))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, dropout_p):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                    nn.BatchNorm2d(out_channels)
            )
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.dropout1(self.conv1(input))))
        input = nn.ReLU()(self.bn2(self.dropout2(self.conv2(input))))
        input = nn.ReLU()(self.bn3(self.dropout3(self.conv3(input))))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, dropout_p=0, useBottleneck=False, outputs=8):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], dropout_p=dropout_p, downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], dropout_p=dropout_p, downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], dropout_p=dropout_p, downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), resblock(filters[2], filters[2], dropout_p=dropout_p, downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], dropout_p=dropout_p, downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), resblock(filters[3], filters[3], dropout_p=dropout_p, downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], dropout_p=dropout_p, downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), resblock(filters[4], filters[4], dropout_p=dropout_p, downsample=False))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = flatten(input, start_dim=1)
        input = self.fc(input)

        return input