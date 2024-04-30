import time

import torch.optim as optim
from PIL import Image
from imagen_pytorch import Unet, Imagen
from extract.getim import load_image
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

def imshow(tensor, title=None):
    # we clone the tensor to not do changes on it
    # remove the fake batch dimension
    image = unloader(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated




torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.ToTensor(),  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
])  # 来自ImageNet的mean和variance

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

def imgsave(image,pa):
    tensor=image.permute(1,2,0)
    print(tensor.shape)
    cv2.imwrite(pa,tensor.cpu().numpy()*255)

trainer = ImagenTrainer(imagen).cuda()
trainer.load("./checkpoint.pt")
model = vggish.WLC(urls="", pretrained=False).cuda()
model.load_state_dict(torch.load("wlc.pt").state_dict())
model.eval()
trainer.eval()
# 18,63,180,327,397,450,674,777,849,1439,1645,

#input the audio name of the test data
list2=["test1","test2","test3"]
for i in list2:
    fmusic = model("./testaudio/"+i+".wav", 16000)["embedding"]
    img=trainer.sample(text_embeds=fmusic,batch_size = 1,cond_scale = 1.)
    img=img.squeeze(0)
    imgsave("./testresult/"+i+".png")