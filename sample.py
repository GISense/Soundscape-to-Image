import time

import torch.optim as optim
from PIL import Image
from imagen_pytorch import Unet, Imagen
from data.getim import load_image
from torchvggish import vggish
import torch
from torchvision import models
from torchvision import transforms
import os
import numpy as np
import cv2
from imagen_pytorch import ImagenTrainer
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import calendar
import time

def gtim():
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    return time_stamp


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()
def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles
def imshow(tensor, title=None):
    # we clone the tensor to not do changes on it
    # remove the fake batch dimension
    image = unloader(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def imgsave(image,pa):
    tensor=image.permute(1,2,0)
    print(tensor.shape)
    cv2.imwrite(pa,tensor.cpu().numpy()*255)
    return
def main(args):
    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ]) 

    unet1 = Unet(
        dim = 128,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
    )

    unet2 = Unet(
        dim = 128,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    imagen = Imagen(
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        timesteps = 256,
        cond_drop_prob = 0.1,
        lowres_sample_noise_level = 0.1,
        random_crop_sizes = (None, 64),
        condition_on_text=True
    )


    trainer = ImagenTrainer(imagen).to(device)
    trainer.load(args.unet_ckpt)

    model = vggish.WLC(urls="", pretrained=False).to(device)
    model.load_state_dict(torch.load(args.audio_encoder_ckpt).state_dict())
    model.eval()
    trainer.eval()

    files_entire = getAllFiles(args.test_audio_path)
    files = [i.split(".")[0] for i in files_entire]

    for i in files:
        fmusic = model(f'{args.test_audio_path}/{i}.wav', 16000)["embedding"]
        img=trainer.sample(text_embeds=fmusic,batch_size = 1,cond_scale = args.cond_scale)
        img=img.squeeze(0)
        imgsave(img , f'{args.test_image_path}/{i}.png')
        imgsave("./testresult/"+i+"_generated.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-enconder-ckpt', type=str, default= None, required=False, help="path to the checkpoint")
    parser.add_argument('--unet-ckpt', type=str, default= None, required=False, help="path to the checkpoint")
    parser.add_argument('--test-audio-path', type=str, default= './data/audio', required=False, help="path to the dataset")
    parser.add_argument('--test-image-path', type=str, default= './data/image', required=False, help="path to the dataset")
    parser.add_argument('--cond-scale', type=float, default= 1., required=False, help="path to the dataset")
    args = parser.parse_args()
    main(args)