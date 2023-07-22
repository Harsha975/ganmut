import os
from pathlib import Path
import sys
import dlib
import matplotlib.pyplot as plt
import torch
import cv2
from imutils.face_utils import rect_to_bb
import numpy as np

sys.path.append('../')
from models.model_linear_2d import Generator as Generator_l2
from models.model_gaussian_2d import Generator as Generator_g2

class GANmut:

    def __init__(self, G_path, model='linear', g_conv_dim=64, c_dim=7, g_repeat_num=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)
        self.model = model

        if self.model == 'linear':
            self.G = Generator_l2(self.device, g_conv_dim, c_dim, g_repeat_num)

        elif self.model == 'gaussian':
            self.G = Generator_g2(self.device, g_conv_dim, c_dim, g_repeat_num)

        else:
            raise ValueError("choose either model='linear' or model='gaussian'")

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G.to(self.device)
        self.detector = dlib.get_frontal_face_detector()

    def emotion_edit(self, img_path, x=None, y=None, theta=None, rho=None, save=False):

        if self.model == 'linear':
            assert (rho is not None) or (theta is not None), 'if model is linear you must provide rho and theta'
        else:
            assert (x is not None) and (y is not None), 'if model is gaussian you must provide x and y'

        img = cv2.imread(img_path, 1)  # BGR
        img_rgb = img[:, :, [2, 1, 0]]
        
        # extract face
        det = self.detector(img, 1)[0]
        (xx, yy, w, h) = rect_to_bb(det)
        face = cv2.resize(img[yy:yy + h, xx:xx + w], (128, 128))

       

        # adapt image format for G
        face = face.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
        face = (face / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        face = torch.from_numpy(face).float().unsqueeze(0).to(self.device)

        # edit emotion

        with torch.no_grad():

            if self.model == 'linear':
                mode = 'manual_selection'
                expr = (torch.tensor([np.cos(theta), np.sin(theta)]) * rho).to(self.device).float()
                face_g = self.G(face, None, None, mode=mode, manual_expr=expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5
            else:
                expr = torch.Tensor([x, y]).unsqueeze(0).to(self.device)
                face_g = self.G(face, expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5

        face_g = face_g.transpose(0, 2).transpose(0, 1).detach().cpu().numpy()

        # insert edited face in original image
        img_rgb[yy:yy + h, xx:xx + w] = cv2.resize(face_g, (w, h)) * 255
        return img_rgb