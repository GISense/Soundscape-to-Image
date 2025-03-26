import torch.optim as optim
import argparse
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


torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.ToTensor(),
]) 

def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        random_crop_sizes = (None, 64)
    )

    trainer = ImagenTrainer(imagen,cosine_decay_max_steps = 1500000,warmup_steps = 7500,lr=args.lr_unet).to(device)
    if args.continue_unet_ckpt:
        trainer.load(args.continue_unet_ckpt)

    model = vggish.WLC(urls="", pretrained=False).to(device)
    model.load_state_dict(torch.load(args.pre_trained_audio_encoder).state_dict())
    model.eval()

    epo= args.epochs
    p= args.batch_size
    files = getAllFiles(args.train_image_path)

    outloss=0

    for k in range(epo):
        for nm in range(0, len(files) + 1 - p, p):
            file_ext0 = os.path.splitext(files[nm])
            front0, ext0 = file_ext0
            fmusic = model(args.train_audio_path + str(front0) + ".wav", 16000)["embedding"]
            fcontent = load_image(args.train_image_path + str(front0) + ".jpg", transform, shape=[256, 256])

            for ni in range(1,p):
                file_ext = os.path.splitext(files[nm+ni])
                front, ext = file_ext
                content = load_image(args.train_image_path + str(front) + ".jpg", transform, shape=[256, 256])
                fcontent = torch.cat((fcontent, content), -4)

                music= model(args.train_audio_path + str(front) + ".wav",16000)["embedding"]
                fmusic = torch.cat((fmusic, music), -3)

            loss = trainer(fcontent, text_embeds = fmusic, unet_number = args.train_unet_number, max_batch_size = p)
            trainer.update(unet_number = args.train_unet_number)
            print(loss)
            outloss=outloss+loss
        outloss=outloss

        print("epoch"+str(k)+" "+" loss: "+str(outloss))

        outloss=0

        if k % args.save_every == 0:
            trainer.save(f"{args.checkpoint_path}/imagen_{args.train_unet_number}_{k}_epochs.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-path', type=str, default= './data/image/', required=False, help="path to the dataset")
    parser.add_argument('--train-audio-path', type=str, default= './data/audio/', required=False, help="path to the dataset")
    parser.add_argument('--pre-trained-audio-encoder', type=str, default= './wlc.pt', required=False, help="path to the pre-trained audio encoder")
    parser.add_argument('--checkpoint-path', type=str, default= './checkpoints', required=False, help="path to the checkpoint")
    parser.add_argument('--batch-size', type=int, default= 5, required=False, help="batch size")
    parser.add_argument('--epochs', type=int, default= 30, required=False, help="number of epochs")
    parser.add_argument('--lr-unet', type=float, default= 1e-4, required=False, help="learning rate")
    parser.add_argument('--continue-unet-ckpt', type=str, default= None, required=False, help="continue training from a checkpoint")
    parser.add_argument('--train-unet-number', type=int, choices=[1, 2], default=1, required=True, help="unet number")
    parser.add_argument('--save-every', type=int, default= 1, required=False, help="save every n epochs")
    args = parser.parse_args()
    main(args)
