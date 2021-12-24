import numpy as np
import cv2


def mdsi(Ref, Dist, combMethod='sum'):
    assert combMethod != 'sum' or combMethod != 'mult' 'Combination method must be either \'sum\' or \'mult\''
    C1 = 140
    C2 = 55
    C3 = 550

    # keep the result same with matlab conv2() when the kernel size is even
    Ref = cv2.flip(Ref, -1)
    Dist = cv2.flip(Dist, -1)

    Ref = Ref.astype(np.float)
    Dist = Dist.astype(np.float)
    H, W, _ = Ref.shape
    min_dimension = np.min((H, W))
    f = np.max((1, int(np.round(min_dimension / 256))))
    ave_kernel = np.ones((f, f)) / (f * f)

    aveR1 = cv2.filter2D(Ref[:, :, 2], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveR1 = cv2.flip(aveR1, -1)
    aveR2 = cv2.filter2D(Dist[:, :, 2], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveR2 = cv2.flip(aveR2, -1)
    R1 = aveR1[0:H:f, 0:W:f]
    R2 = aveR2[0:H:f, 0:W:f]

    aveB1 = cv2.filter2D(Ref[:, :, 0], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveB1 = cv2.flip(aveB1, -1)
    aveB2 = cv2.filter2D(Dist[:, :, 0], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveB2 = cv2.flip(aveB2, -1)
    B1 = aveB1[0:H:f, 0:W:f]
    B2 = aveB2[0:H:f, 0:W:f]

    aveG1 = cv2.filter2D(Ref[:, :, 1], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveG2 = cv2.filter2D(Dist[:, :, 1], -1, ave_kernel, borderType=cv2.BORDER_CONSTANT)
    aveG1 = cv2.flip(aveG1, -1)
    aveG2 = cv2.flip(aveG2, -1)
    G1 = aveG1[0:H:f, 0:W:f]
    G2 = aveG2[0:H:f, 0:W:f]

    L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
    L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2
    F = 0.5 * (L1 + L2)

    H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
    H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
    M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
    M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
    dy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3

    IxL1 = cv2.filter2D(L1, -1, dx, borderType=cv2.BORDER_CONSTANT)
    IyL1 = cv2.filter2D(L1, -1, dy, borderType=cv2.BORDER_CONSTANT)
    gR = np.sqrt(IxL1 ** 2 + IyL1 ** 2)

    IxL2 = cv2.filter2D(L2, -1, dx, borderType=cv2.BORDER_CONSTANT)
    IyL2 = cv2.filter2D(L2, -1, dy, borderType=cv2.BORDER_CONSTANT)
    gD = np.sqrt(IxL2 ** 2 + IyL2 ** 2)

    IxF = cv2.filter2D(F, -1, dx, borderType=cv2.BORDER_CONSTANT)
    IyF = cv2.filter2D(F, -1, dy, borderType=cv2.BORDER_CONSTANT)
    gF = np.sqrt(IxF ** 2 + IyF ** 2)

    GS12 = (2 * gR * gD + C1) / (gR ** 2 + gD ** 2 + C1)
    GS13 = (2 * gR * gF + C2) / (gR ** 2 + gF ** 2 + C2)
    GS23 = (2 * gD * gF + C2) / (gD ** 2 + gF ** 2 + C2)
    GS_HVS = GS12 + GS23 - GS13

    CS = (2 * (H1 * H2 + M1 * M2) + C3) / (H1 ** 2 + H2 ** 2 + M1 ** 2 + M2 ** 2 + C3)

    if combMethod == 'sum':
        alpha = 0.6
        GCS = alpha * GS_HVS + (1 - alpha) * CS
    elif combMethod == 'mult':
        gamma = 0.2
        beta = 0.1
        GCS = GS_HVS ** gamma * CS ** beta
    GCS = GCS.flatten('F')
    Q = np.average(np.abs((GCS ** 0.25) - np.average(GCS ** 0.25))) ** 0.25
    return Q
