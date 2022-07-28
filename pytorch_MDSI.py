import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MDSI(nn.Module):
    def __init__(self, combMethod='sum'):
        super(MDSI, self).__init__()
        self.C1 = 140
        self.C2 = 55
        self.C3 = 550
        self.combMethod = combMethod

    def forward(self, ref, dist):
        assert ref.shape == dist.shape
        C, H, W = ref.shape
        min_dimension = np.min((H, W))
        f = np.max((1, int(np.round(min_dimension / 256))))
        avg_kernel = torch.ones((f, f)) / (f * f)

        avgR1 = F.conv2d(ref[:, 0, :, :], avg_kernel, padding='same')
        avgR2 = F.conv2d(dist[:, 0, :, :], avg_kernel, padding='same')
        R1 = avgR1[:, :, 0:H:f, 0:W:f]
        R2 = avgR2[:, :, 0:H:f, 0:W:f]

        avgG1 = F.conv2d(ref[:, 1, :, :], avg_kernel, padding='same')
        avgG2 = F.conv2d(dist[:, 1, :, :], avg_kernel, padding='same')
        G1 = avgG1[:, :, 0:H:f, 0:W:f]
        G2 = avgG2[:, :, 0:H:f, 0:W:f]

        avgB1 = F.conv2d(ref[:, 2, :, :], avg_kernel, padding='same')
        avgB2 = F.conv2d(dist[:, 2, :, :], avg_kernel, padding='same')
        B1 = avgB1[:, :, 0:H:f, 0:W:f]
        B2 = avgB2[:, :, 0:H:f, 0:W:f]

        L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
        L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2
        f = 0.5 * (L1 + L2)

        H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
        H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
        M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
        M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

        dx = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
        dy = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3

        IxL1 = F.conv2d(L1, dx, padding='same')
        IyL1 = F.conv2d(L1, dy, padding='same')
        gR = torch.sqrt(IxL1**2 + IyL1**2)

        IxL2 = F.conv2d(L2, dx, padding='same')
        IyL2 = F.conv2d(L2, dy, padding='same')
        gD = torch.sqrt(IxL2 ** 2 + IyL2 ** 2)

        IxF = F.conv2d(f, dx, padding='same')
        IyF = F.conv2d(f, dy, padding='same')
        gF = torch.sqrt(IxF ** 2 + IyF ** 2)

        GS12 = (2 * gR * gD + self.C1) / (gR ** 2 + gD ** 2 + self.C1)
        GS13 = (2 * gR * gF + self.C2) / (gR ** 2 + gF ** 2 + self.C2)
        GS23 = (2 * gD * gF + self.C2) / (gD ** 2 + gF ** 2 + self.C2)
        GS_HVS = GS12 + GS23 - GS13

        CS = (2 * (H1 * H2 + M1 * M2) + self.C3) / (H1 ** 2 + H2 ** 2 + M1 ** 2 + M2 ** 2 + self.C3)
        if self.combMethod == 'sum':
            alpha = 0.6
            GCS = alpha * GS_HVS + (1 - alpha) * CS
        elif self.combMethod == 'mult':
            gamma = 0.2
            beta = 0.1
            GCS = GS_HVS ** gamma * CS ** beta
        GCS = GCS.flatten(1)
        Q = torch.mean(torch.abs(GCS ** 0.25 - torch.mean(GCS ** 0.25))) ** 0.25
        return Q


if __name__ == '__main__':
    a = torch.ones((1, 1, 5, 5))
    b = a[:, :, 0:5:2, 0:5:2]
    print(b)