import sys

sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import model_big


def predict_cap(
    img_data,
    labels_data,
    gpu_id=0,
    random=False,
    outf="checkpoints/binary_faceforensicspp",
    id=21,
):
    print(labels_data)
    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(outf, "capsule_" + str(id) + ".pt")))
    capnet.eval()

    if gpu_id >= 0:
        vgg_ext.cuda(gpu_id)
        capnet.cuda(gpu_id)
    labels_data[labels_data > 1] = 1
    img_label = labels_data.numpy().astype(np.float64)

    if gpu_id >= 0:
        img_data = img_data.cuda(gpu_id)
        labels_data = labels_data.cuda(gpu_id)

    input_v = Variable(img_data)

    x = vgg_ext(input_v)
    classes, class_ = capnet(x, random=random)

    output_dis = class_.data.cpu()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float64)
    for i in range(output_dis.shape[0]):
        if output_dis[i, 1] >= output_dis[i, 0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0
    return output_pred
