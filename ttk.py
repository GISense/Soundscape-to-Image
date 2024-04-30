import data
from imagen_pytorch import Unet, Imagen
import torch
from torchvision import transforms
from models import imagebind_model
from models.imagebind_model import ModalityType
import torch.nn as nn
from imagen_pytorch import ImagenTrainer
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
def imagebind_out(audio_paths,model):
    # Load data
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings

class encode_audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.link2=nn.Linear(1024,343)
        self.to(self.device)

    def forward(self,embeddings):
        l1=embeddings
        l2=self.link2(embeddings)
        # l3=self.link3(embeddings)
        l3=torch.matmul(l2.transpose(1,2),l1)

        return torch.cat([l1,l3],dim=1)


unet1 = Unet(
    dim = 128,
    cond_dim = 128,
    dim_mults = (1, 2, 3, 4),
    num_resnet_blocks = 3,
    attn_dim_head = 64,
    attn_heads = 8,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True),
    memory_efficient = False,
)


unet2 = Unet(
    dim = 128,
    cond_dim = 128,
    dim_mults = (1, 2, 3, 4),
    num_resnet_blocks = (2, 4, 8, 8),
    attn_dim_head = 64,
    attn_heads = 8,
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True),
)

imagen = Imagen(
    text_embed_dim=1024,
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 256,
    condition_on_text = True,
    cond_drop_prob = 0.1,
    lowres_sample_noise_level=0.1,
    # random_crop_sizes = (None, 64)
)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Instantiate model
model_imageb = imagebind_model.imagebind_huge(pretrained=True)
model_imageb.eval()
model_imageb.to(device)

trainer = ImagenTrainer(
    imagen = imagen,
    cosine_decay_max_steps = 1500000,
    warmup_steps = 7500
).cuda()


trainer.load("./checkpoint.pt")
model1=encode_audio()
model1.load_state_dict(torch.load("wlc2.pt").state_dict())
model1.eval()
trainer.eval()

def imgsave(image,pa):
    tensor=image.permute(1,2,0)
    print(tensor.shape)
    cv2.imwrite(pa,tensor.cpu().numpy()*255)

# lik=["18","63","180","327","397","450","674","777","849","1439","1645","2004","3258","3655","3886","4100","4196","4503","4749","4829","5280","5292","5498","5500","5593","5722","6409","7798","8226","8360"]
lik=["7307","7308","7309"]
for name in lik:
    for li in range(20):
        audio_pat = []
        audio_pat.append("./extract/audio/"+name+".wav")
        imageb_out = imagebind_out(audio_pat, model_imageb)
        fmusic = model1(imageb_out["audio"].unsqueeze(1))
        img=trainer.sample(text_embeds=fmusic,batch_size = 1,cond_scale = 1)
        img=img.squeeze(0)
        imgsave(img, "./out3/"+name+"-"+str(li)+".png")
        # plt.figure()
        # imshow(img, title='Target1 Image')
