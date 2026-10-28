#!/usr/bin/env python3
"""
Convolutional Forward Propagation Module
"""
import numpy as np


def conv_forward(A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural net
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    out_h = int((h_prev + 2 * ph - kh) / sh) + 1
    out_w = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            slice_A = A_pad[:, v_start:v_end, h_start:h_end, :]

            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    slice_A * W[:, :, :, k], axis=(1, 2, 3)
                ) + b[0, 0, 0, k]

    return Z