import torch.optim as optim

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


torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.ToTensor(),  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
])  # 来自ImageNet的mean和variance

def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles

# unet for imagen

#unet1 = NullUnet()  # add a placeholder "null" unet for the base unet

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

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 256,
    cond_drop_prob = 0.1,
    lowres_sample_noise_level = 0.1,
    random_crop_sizes = (None, 64)
)

trainer = ImagenTrainer(imagen,cosine_decay_max_steps = 1500000,warmup_steps = 7500,lr=1e-4).cuda()
trainer.load("./checkpoint.pt")


# mock images (get a lot of this) and text encodings from large T5

# mock images (get a lot of this) and text encodings from large T5
model = vggish.WLC(urls="", pretrained=False).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5,
                       betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
model.load_state_dict(torch.load("wlc.pt").state_dict())
model.eval()
# model.train()
epo=31
p=5
files = getAllFiles("./extract/audio")

outloss=0

for k in range(epo):
    for nm in range(0, len(files) + 1 - p, p):
        #for i in (1, 2):
        file_ext0 = os.path.splitext(files[nm])
        front0, ext0 = file_ext0
        fmusic = model("./extract/audio/" + str(front0) + ".wav", 16000)["embedding"]
        fcontent = load_image("./extract/image/" + str(front0) + ".jpg", transform, shape=[256, 256])

        for ni in range(1,p):
            file_ext = os.path.splitext(files[nm+ni])
            front, ext = file_ext
            content = load_image("./extract/image/" + str(front) + ".jpg", transform, shape=[256, 256])
            fcontent = torch.cat((fcontent, content), -4)

            music= model("./extract/audio/" + str(front) + ".wav",16000)["embedding"]
            fmusic = torch.cat((fmusic, music), -3)

        loss = trainer(fcontent, text_embeds=fmusic, unet_number=2,max_batch_size = p)
        trainer.update(unet_number = 2)
        # optimizer.step()
        # print(optimizer.state)
        # optimizer.zero_grad()
        print(loss)
        outloss=outloss+loss
        #print("unet"+str(i)+" "+str(loss))

    outloss=outloss

    print("epoch"+str(k)+" "+" loss: "+str(outloss))

    outloss=0

    if (k == 0):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 2):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 6):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 10):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 15):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 20):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 25):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
    if (k == 30):
        trainer.save('./checkpoint.pt')
        torch.save(model, "wlc.pt")
