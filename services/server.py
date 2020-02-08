import io
import os
import cv2
import copy
import grpc
import time
import torch
import numpy as np

from PIL import Image
from concurrent import futures

from services import image_transfer_pb2
from services import image_transfer_pb2_grpc

from util import util
from models.models import create_model
from data.base_dataset import get_transform
from options.test_options import TestOptions


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def get_Images(img_1, img_2):
    return image_transfer_pb2.Images(img_1=img_1, img_2=img_2)


class ImageGenerator(image_transfer_pb2_grpc.ImageTransferServicer):
    def __init__(self, opt):
        self.transform = get_transform(opt)
        self.model = create_model(opt)

        opt2 = copy.deepcopy(opt)
        opt2.name = opt.name_pix2pix
        opt2.model = opt.model_pix2pix
        opt2.ndisc_out_filters = 1
        opt2.ndres_down = 4
        opt2.ngres = 16
        opt2.ndres = 16
        opt2.ngf = 64
        opt2.ndf = 64
        opt2.spectral_G = True
        opt2.spectral_D = True
        opt2.norm_G = 'instance'
        opt2.norm_D = 'instance'
        opt2.checkpoints_dir = opt2.checkpoints_dir_pix2pix
        opt2.res_op = 'add'
        opt2.which_epoch = opt.which_epoch_pix2pix
        opt2.shadow = True
        opt2.nz = 8
        opt2.input_nc = 3
        opt2.loadSize = 256
        opt2.fineSize = 256

        self.transform_color = get_transform(opt2)
        self.model_color = create_model(opt2)

    def preprocess(self, image, label=0):
        cv2_scribble = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_scribble = Image.fromarray(cv2_scribble)

        A = self.transform(pil_scribble)
        A = A.resize_(1, opt.input_nc, 128, 128)
        A = A.expand(opt.num_interpolate, opt.input_nc, 128, 128)
        B = A
        label = torch.LongTensor([label])
        label = label.expand(opt.num_interpolate)
        data = {'A': A, 'A_sparse': A, 'A_mask': A,
                'B': B, 'A_paths': '', 'B_paths': '', 'label': label}
        return data

    def preprocess_color(self, label=0):
        pil_scribble = Image.open(
            'imgs/test_0_L_fake_B_inter.png').convert('RGB')
        pil_scribble = pil_scribble.resize((256, 256), Image.ANTIALIAS)
        A = self.transform_color(pil_scribble)
        A = A.resize_(1, 3, 256, 256)
        B = A
        label = torch.LongTensor([[label]])
        data = {'A': A, 'A_sparse': A, 'A_mask': A,
                'B': B, 'A_paths': '', 'B_paths': '', 'label': label}
        return data

    def GetGenerateImage(self, request, context):
        image_data = request.data
        label_id = int(request.label)
        shadow = int(request.shadow)
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (256, 256, 3))

        data = self.preprocess(image, label_id)
        self.model.set_input(data)
        visuals = self.model.get_latent_noise_visualization()
        image_dir = './imgs'
        for label, image_numpy in visuals.items():
            image_name = 'test_%s.png' % (label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

        if shadow:
            cv2_img = cv2.imread('./imgs/test_fake_B_shadow.png')
        else:
            cv2_img = cv2.imread('imgs/test_0_L_fake_B_inter.png')
        cv2_img = cv2.resize(cv2_img, (256, 256))

        data = self.preprocess_color(label_id)
        self.model_color.set_input(data)
        self.model_color.test()
        visuals = self.model_color.get_current_visuals()

        for label, image_numpy in visuals.items():
            image_name = 'test_color_%s.png' % (label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

        res_img = np.array(
            Image.open('imgs/test_color_fake_B.png'), dtype=np.uint8)

        return get_Images(cv2_img.tobytes(), res_img.tobytes())

    def RandomizeNoise(self, request, context):
        self.model.randomize_noise()
        return image_transfer_pb2.Empty()


def start(server, opt):
    image_transfer_pb2_grpc.add_ImageTransferServicer_to_server(
        ImageGenerator(opt), server)
    server.add_insecure_port('[::]:{}'.format(opt.port))
    server.start()
    print('Service start on port %d' % opt.port)


def serve(opt):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    start(server, opt)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.loadSize = 256

    serve(opt)
