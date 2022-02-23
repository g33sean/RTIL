from tqdm import tqdm
import numpy as np
import argparse
import random
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from shutil import copy
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from progan_modules import Generator, Discriminator

def calculate_gradient_penalty(model,step,alpha, real_images, fake_images, lambd ,device):
    """Calculates the gradient penalty loss for WGAN GP"""
    eps = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    eps = eps.expand_as(real_images)
    x_hat = eps * real_images + (1 - eps) * fake_images.detach()
    x_hat.requires_grad = True
    px_hat = model(x_hat,step=step,alpha=alpha)
    grad = torch.autograd.grad(outputs = px_hat.sum(),
                                    inputs = x_hat, 
                                    create_graph=True
                                    )[0]
    grad_norm = grad.view(real_images.size(0), -1).norm(2, dim=1)
    gradient_penalty = lambd * ((grad_norm  - 1)**2).mean()
    return gradient_penalty
    
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def imagefolder_loader(path):
    def loader(transform,batch_size):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,drop_last=False)
        return data_loader
    return loader


def sample_data(dataloader,batch_size, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size+int(image_size*0.2)+1),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform,batch_size)

    return loader


def train(args,generator, discriminator,g_optimizer,d_optimizer,loader):
    step = args.init_step # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    data_loader = sample_data(loader,args.batch_size, 4 * 2 ** step)
    dataset = iter(data_loader)


    total_iter=args.total_iter
    pbar = range(total_iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

  
    alpha = args.alpha


    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    if args.ckpt is not None:
        iteration=args.iteration
    else:
        iteration=0

    for idx in pbar:
        i=idx + args.start_iter

        if i>args.total_iter:
            print("Done!")
            break
        discriminator.zero_grad()

        alpha = min(1, (2/(total_iter//7)) * iteration)


        if iteration > total_iter//7:
            print('*'*100)
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            if step ==0:
                data_loader = sample_data(loader,args.batch_size, 4 * 2 ** step)
            if step==1:
                batch_size=32
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)
            if step==2:
                batch_size=32
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)
            if step==3:
                batch_size=16
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)
            if step==4:
                batch_size=8
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)
            if step==5:
                batch_size=6
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)
            if step==6:
                batch_size=6
                data_loader = sample_data(loader,batch_size, 4 * 2 ** step)



            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image,_= next(dataset)

        iteration += 1

        ### 1. train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(device)
      

        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        if args.rtil ==True:
            gen_z = torch.randn(b_size, input_code_size).to(device)
            gen_z1 = torch.randn(args.z_number[0]*b_size, input_code_size).to(device)
            gen_z2 = torch.randn(args.z_number[1]*b_size, input_code_size).to(device)

            fake_image = generator(gen_z,gen_z1,gen_z2, step=step, alpha=alpha,rtil=0)
            fake_image1=generator(gen_z,gen_z1,gen_z2, step=step, alpha=alpha,rtil=1)
            fake_image2=generator(gen_z,gen_z1,gen_z2, step=step, alpha=alpha,rtil=2)

            fake_predict1 = discriminator(fake_image.detach(), step=step, alpha=alpha)
            fake_predict2 = discriminator(fake_image1.detach(), step=step, alpha=alpha)
            fake_predict3 = discriminator(fake_image2.detach(), step=step, alpha=alpha)
            fake_predict = 1/3*(fake_predict1.mean() + fake_predict2.mean() + fake_predict3.mean())
            fake_predict.backward(one)


            grad_penalty1=calculate_gradient_penalty(discriminator,step,alpha,real_image,fake_image,lambd=10,device=device)
            grad_penalty2=calculate_gradient_penalty(discriminator,step,alpha,real_image,fake_image1,lambd=10,device=device)
            grad_penalty3=calculate_gradient_penalty(discriminator,step,alpha,real_image,fake_image2,lambd=10,device=device)

            grad_penalty=(1/3)*(grad_penalty1+grad_penalty2+grad_penalty3)

        else:
            gen_z = torch.randn(b_size, input_code_size).to(device)
            gen_z1 = torch.randn(args.z_number[0]*b_size, input_code_size).to(device)
            gen_z2 = torch.randn(args.z_number[1]*b_size, input_code_size).to(device)

            fake_image = generator(gen_z,gen_z1,gen_z2, step=step, alpha=alpha,rtil=0)
         

            fake_predict = discriminator(fake_image.detach(), step=step, alpha=alpha)
            fake_predict=fake_predict.mean()
            
            fake_predict.backward(one)

            grad_penalty=calculate_gradient_penalty(discriminator,step,alpha,real_image,fake_image,lambd=10,device=device)


        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()

            if args.rtil==True:
                predict = discriminator(fake_image, step=step, alpha=alpha)
                predict1 = discriminator(fake_image1, step=step, alpha=alpha)
                predict2 = discriminator(fake_image2, step=step, alpha=alpha)

                loss = -1/3*(predict.mean()+predict1.mean()+predict2.mean())
            
            else:
                predict = discriminator(fake_image, step=step, alpha=alpha)
                loss=-predict.mean()


            gen_loss_val += loss.item()


            loss.backward()
            g_optimizer.step()
            accumulate(g_ema, generator)

        if (i + 1) % 1000 == 0 or i==0:
            if args.rtil==True:
                with torch.no_grad():
                    z=torch.randn(15, input_code_size).to(device)
                    z1=torch.randn(args.z_number[0]*15, input_code_size).to(device)
                    z2=torch.randn(args.z_number[1]*15, input_code_size).to(device)
                    images = g_ema(z,z1,z2, step=step, alpha=alpha,rtil=0).data.cpu()
                    images1=g_ema(z,z1,z2, step=step, alpha=alpha,rtil=1).data.cpu()
                    images2=g_ema(z,z1,z2, step=step, alpha=alpha,rtil=2).data.cpu()
                    utils.save_image(
                        images,
                        f'rtil/sample/{str(i + 1).zfill(6)}.png',
                        nrow=10,
                        normalize=True,
                        range=(-1, 1))
                    utils.save_image(
                        images1,
                        f'rtil/sample1/{str(i + 1).zfill(6)}.png',
                        nrow=10,
                        normalize=True,
                        range=(-1, 1))
                    utils.save_image(
                        images2,
                        f'rtil/sample2/{str(i + 1).zfill(6)}.png',
                        nrow=10,
                        normalize=True,
                        range=(-1, 1))
            else: 
                 with torch.no_grad():
                    z=torch.randn(15, input_code_size).to(device)
                    z1=torch.randn(args.z_number[0]*15, input_code_size).to(device)
                    z2=torch.randn(args.z_number[1]*15, input_code_size).to(device)
                    images = g_ema(z,z1,z2, step=step, alpha=alpha,rtil=0).data.cpu()
                    utils.save_image(
                        images,
                        f'van/sample/{str(i + 1).zfill(6)}.png',
                        nrow=10,
                        normalize=True,
                        range=(-1, 1))
 
        if (i+1) % 1000 == 0 or i==0:
           
            try:
                if args.rtil==True:
                    torch.save(
                        {
                            "g": generator.state_dict(),
                            "d": discriminator.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optimizer": g_optimizer.state_dict(),
                            "d_optimizer": d_optimizer.state_dict(),
                            "alpha":alpha,
                            "step":step,
                            "iteration":iteration,
                            "args": args,
                        },
                        f"rtil/checkpoint/{str(i).zfill(6)}.pt",
                    )
                else:
                     torch.save(
                        {
                            "g": generator.state_dict(),
                            "d": discriminator.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optimizer": g_optimizer.state_dict(),
                            "d_optimizer": d_optimizer.state_dict(),
                            "alpha":alpha,
                            "step":step,
                            "iteration":iteration,
                            "args": args,
                        },
                        f"van/checkpoint/{str(i).zfill(6)}.pt",
                    )
            except:
                pass

        if (i+1)%500 == 0:
            state_msg = (f'{i + 1}; G: {gen_loss_val/(500//n_critic):.3f}; D: {disc_loss_val/500:.3f};'
                f' Grad: {grad_loss_val/500:.3f}; Alpha: {alpha:.3f}')
            

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--rtil', default=False, action="store_true", help='if you want to train with RTIL training if set false vanilla training ')
    parser.add_argument('--path', type=str,default='./ffhq', help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument("--ckpt",type=str,default=None,help="path to the checkpoints to resume training")
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=512, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=256, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=32, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--alpha', type=int, default=0, help='fade in alpha')
    parser.add_argument('--init_step', type=int, default=0, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=500000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=True, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--z_number',type=list ,default=[10,20],help='No. of latent variables')
    
    args = parser.parse_args()

    #print(str(args))


 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic
    args.start_iter=0
 
    generator = Generator(args,in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_ema = Generator(args,in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    
    ## you can directly load a pretrained model here
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            print(args.start_iter)

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optimizer.load_state_dict(ckpt["g_optimizer"])
        d_optimizer.load_state_dict(ckpt["d_optimizer"])
        args.alpha=ckpt["alpha"]
        args.init_step=ckpt["step"]
        args.iteration=ckpt["iteration"]
       
    
 
    g_ema.train(False)

    
    accumulate(g_ema, generator, 0)

    loader = imagefolder_loader(args.path)

    train(args,generator, discriminator,g_optimizer,d_optimizer, loader)
