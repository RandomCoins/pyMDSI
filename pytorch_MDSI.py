import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class MDSI(nn.Module):
    def __init__(self, combMethod='sum'):
        super(MDSI, self).__init__()
        self.C1 = 140
        self.C2 = 55
        self.C3 = 550
        self.combMethod = combMethod

    def forward(self, ref, dist):
        assert ref.shape == dist.shape
        _, C, H, W = ref.shape
        min_dimension = np.min((H, W))
        f = np.max((1, int(np.round(min_dimension / 256))))
        avg_kernel = torch.ones((3, 1, f, f)) / (f * f)
        avgRef = F.conv2d(ref, avg_kernel, padding='same', groups=C)
        avgDist = F.conv2d(dist, avg_kernel, padding='same', groups=C)

        R1 = avgRef[:, 0, 0:H:f, 0:W:f]
        R2 = avgDist[:, 0, 0:H:f, 0:W:f]

        G1 = avgRef[:, 1, 0:H:f, 0:W:f]
        G2 = avgDist[:, 1, 0:H:f, 0:W:f]

        B1 = avgRef[:, 2, 0:H:f, 0:W:f]
        B2 = avgDist[:, 2, 0:H:f, 0:W:f]

        L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
        L1 = L1.unsqueeze(0)
        L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2
        L2 = L2.unsqueeze(0)
        f = 0.5 * (L1 + L2)

        H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
        H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
        M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
        M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

        dx = torch.tensor([[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]])
        dx = dx.unsqueeze(0).unsqueeze(0)
        dy = torch.tensor([[1/3, 1/3, 1/3], [0, 0, 0], [-1/3, -1/3, -1/3]])
        dy = dy.unsqueeze(0).unsqueeze(0)

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


def main():
    mdsi = MDSI()
    img1 = cv2.imread('F:\\project\\SourceImage\\L12_BeachHouse.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0)
    img1 = img1.permute(0, 3, 1, 2).contiguous()
    img2 = cv2.imread('G:\\mydataset\\1_L12_BeachHouse_bpg_0.0900.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)
    img2 = img2.permute(0, 3, 1, 2).contiguous()
    res = mdsi(img1, img2)
    print(res.item())


if __name__ == '__main__':
    main()
