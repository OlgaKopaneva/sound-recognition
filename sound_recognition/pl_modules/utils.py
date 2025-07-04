import numpy as np
import torch


class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """x, y — батчи тензоров"""
        batch_size = x.size(0)
        le = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size)

        x_mix = le * x + (1 - le) * x[index]
        y_mix = le * y + (1 - le) * y[index]
        return x_mix, y_mix


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top : top + h, left : left + w, :] = c

        return input_img

    return eraser
