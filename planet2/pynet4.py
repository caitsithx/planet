import torch
import torch.nn as nn
import torch.functional as F


def make_conv_bn_prelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(out_channels),
    ]


class PyNet_4(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(PyNet_4, self).__init__()
        in_channels, height, width = in_shape
        stride = 1

        self.preprocess = nn.Sequential(
            *make_conv_bn_prelu(in_channels, 16,
                                kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(16, 16, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(16, 16, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(16, 16, kernel_size=1, stride=1, padding=0),
        )  # 128

        self.down1 = nn.Sequential(
            *make_conv_bn_prelu(16, 32, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(32, 32, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_prelu(32, 64, kernel_size=1, stride=1, padding=0),
        )  # 128
        self.down1_short = nn.Conv2d(
            16, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.down2 = nn.Sequential(
            *make_conv_bn_prelu(64, 64,  kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_prelu(64, 128, kernel_size=1, stride=1, padding=0),
        )  # 64
        self.down2_short = nn.Conv2d(
            64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.down3 = nn.Sequential(
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(128, 128, kernel_size=3,
                                stride=1, padding=1, groups=16),
            *make_conv_bn_prelu(128, 256, kernel_size=1, stride=1, padding=0),
        )  # 32
        self.down3_short = nn.Conv2d(
            128, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.down4 = nn.Sequential(
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(256, 256, kernel_size=3,
                                stride=1, padding=1, groups=16),
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
        )  # 16
        self.down4_short = None  # nn.Identity()

        self.down5 = nn.Sequential(
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(256, 256, kernel_size=3,
                                stride=1, padding=1, groups=16),
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
        )  # 8
        self.down5_short = None  # nn.Identity()

        self.up4 = nn.Sequential(
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(256, 256, kernel_size=3,
                                stride=1, padding=1, groups=16),
            *make_conv_bn_prelu(256, 256, kernel_size=1, stride=1, padding=0),
        )  # 16
        self.up4_short = None  # nn.Identity()

        self.up3 = nn.Sequential(
            *make_conv_bn_prelu(256, 128, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(128, 128, kernel_size=3,
                                stride=1, padding=1, groups=16),
            *make_conv_bn_prelu(128, 128, kernel_size=1, stride=1, padding=0),
        )  # 32
        #self.up3_short =  nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.up2 = nn.Sequential(
            *make_conv_bn_prelu(128, 64, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0),
        )  # 64
        #self.up2_short =  nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.up1 = nn.Sequential(
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0),
            *make_conv_bn_prelu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_prelu(64, 64, kernel_size=1, stride=1, padding=0),
        )  # 128
        # self.up1_short =  None # nn.Identity()

        self.cls1 = nn.Sequential(
            *make_linear_bn_relu(64, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls2 = nn.Sequential(
            *make_linear_bn_relu(64, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3 = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4 = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls5 = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )

        self.cls1a = nn.Sequential(
            *make_linear_bn_relu(64, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls2a = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3a = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4a = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        out = self.preprocess(x)  # 128

        down1 = self.down1(out)  # 128
        out = F.max_pool2d(down1, kernel_size=2, stride=2)  # 64

        down2 = self.down2(out) \
            + make_shortcut(out, self.down2_short)      # 64
        out = F.max_pool2d(down2, kernel_size=2, stride=2)  # 32
        flat2a = make_max_flat(out)

        down3 = self.down3(out) \
            + make_shortcut(out, self.down3_short)      # 32
        out = F.max_pool2d(down3, kernel_size=2, stride=2)  # 16
        flat3a = make_max_flat(out)

        down4 = self.down4(out) \
            + make_shortcut(out, self.down4_short)      # 16
        out = F.max_pool2d(down4, kernel_size=2, stride=2)  # 8
        flat4a = make_max_flat(out)

        out = self.down5(out)
        #out   = self.down5(out) \
        #         + make_shortcut(out, self.down5_short)      #  8
        # F.dropout(out, p=0.10,training=self.training,inplace=False)
        flat5 = out
        flat5 = make_max_flat(flat5)

        up4 = F.upsample_bilinear(out, scale_factor=2)      # 16
        up4 = up4 + down4                                  # 16   #torch.cat()
        out = self.up4(up4)                                # 16
        # F.dropout(out, p=0.10,training=self.training,inplace=False)
        flat4 = out
        flat4 = make_max_flat(flat4)

        up3 = F.upsample_bilinear(out, scale_factor=2)      # 32
        up3 = up3 + down3                                  # 32
        out = self.up3(up3)                                # 32
        # F.dropout(out, p=0.10,training=self.training,inplace=False)
        flat3 = out
        flat3 = make_max_flat(flat3)

        up2 = F.upsample_bilinear(out, scale_factor=2)      # 64
        up2 = up2 + down2                                  # 64
        out = self.up2(up2)                                # 64
        # F.dropout(out, p=0.10,training=self.training,inplace=False)
        flat2 = out
        flat2 = make_max_flat(flat2)

        up1 = F.upsample_bilinear(out, scale_factor=2)  # 128
        up1 = up1 + down1  # 128
        out = self.up1(up1)  # 128
        # F.dropout(out, p=0.10,training=self.training,inplace=False)
        flat1 = out
        flat1 = make_max_flat(flat1)

        # flat1 = F.dropout(flat1,p=0.1)
        # flat2 = F.dropout(flat2,p=0.1)
        # flat3 = F.dropout(flat3,p=0.1)
        # flat4 = F.dropout(flat4,p=0.1)
        # flat5 = F.dropout(flat5,p=0.1)

        logit1 = self.cls1(flat1).unsqueeze(2)
        logit2 = self.cls2(flat2).unsqueeze(2)
        logit3 = self.cls3(flat3).unsqueeze(2)
        logit4 = self.cls4(flat4).unsqueeze(2)
        logit5 = self.cls5(flat5).unsqueeze(2)

        logit2a = self.cls2a(flat2a).unsqueeze(2)
        logit3a = self.cls3a(flat3a).unsqueeze(2)
        logit4a = self.cls4a(flat4a).unsqueeze(2)

        logit = torch.cat((logit1, logit2, logit3, logit4,
                           logit5, logit2a, logit3a, logit4a), dim=2)
        logit = F.dropout(logit, p=0.15, training=self.training)
        logit = logit.sum(2)
        logit = logit.view(logit.size(0), logit.size(1))  # unsqueeze(2)
        prob = F.sigmoid(logit)

        return logit, prob
