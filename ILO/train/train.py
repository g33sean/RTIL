import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import torchvision.datasets as dset



from dataset import MultiResolutionDataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(path_lengths,mean_path_length, decay=0.01):

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    '''
        Trains a family of Generative model with the intend of being used for ILO inversion method downstream for solving inverse problems. 
        If args.rtil==True will train a family of generative models otherwise the model will be trained with a standard Gan objective.

    '''

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    if args.rtil==True:
        mean_path_length1 = 0
        mean_path_length2 = 0
        mean_path_length3 = 0
        mean_path_length4 = 0
        mean_path_length5 = 0
    else:
        mean_path_length1 = 0


    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(
            args.ada_target, args.ada_length, 8, device)


    if args.rtil==True:
        sample_z = torch.randn(args.n_sample, args.latent, device=device)
        sample_z1 = torch.randn(args.n_sample, args.latent, device=device)
        sample_z2 = torch.randn(args.n_sample, args.latent, device=device)
        sample_z3 = torch.randn(args.n_sample, args.latent, device=device)
        sample_z4 = torch.randn(args.n_sample, args.latent, device=device)
    else:
        sample_z = torch.randn(args.n_sample, args.latent, device=device)

  


    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img[0].to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if args.rtil==True:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise3 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise4 = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)


        if args.rtil==True:
            fake_img,_ = generator(noise, rtil=0)
            fake_img1,_ = generator(noise1, rtil=1)
            fake_img2,_ = generator(noise2, rtil=2)
            fake_img3,_ = generator(noise3, rtil=3)
            fake_img4,_ = generator(noise4,rtil=4)
        else:
            fake_img,_ = generator(noise, rtil=0)




        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        if args.rtil==True:
            fake_pred = discriminator(fake_img)
            fake_pred1 = discriminator(fake_img1)
            fake_pred2 = discriminator(fake_img2)
            fake_pred3 = discriminator(fake_img3)
            fake_pred4 = discriminator(fake_img4)

        else:
            fake_pred = discriminator(fake_img)
        



        real_pred = discriminator(real_img_aug)

        if args.rtil==True:
            d_loss1 = d_logistic_loss(real_pred, fake_pred)
            d_loss2 = d_logistic_loss(real_pred, fake_pred1)
            d_loss3 = d_logistic_loss(real_pred, fake_pred2)
            d_loss4 = d_logistic_loss(real_pred, fake_pred3)
            d_loss5 = d_logistic_loss(real_pred, fake_pred4)


            d_loss = 1/5 * (d_loss1 + d_loss2 + d_loss3 +
                        d_loss4 + d_loss5)
        else:
            d_loss = d_logistic_loss(real_pred, fake_pred)




      
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)


        if args.rtil==True:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise3 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise4 = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)



        if args.rtil==True:
            fake_img, _ = generator(noise, rtil=0)
            fake_img1, _ = generator(noise1, rtil=1)
            fake_img2, _ = generator(noise2, rtil=2)
            fake_img3, _ = generator(noise3, rtil=3)
            fake_img4, _ = generator(noise4, rtil=4)
        else:
            fake_img, _ = generator(noise, tail=0)



        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)


        if args.rtil==True:

            fake_pred = discriminator(fake_img)
            fake_pred1 = discriminator(fake_img1)
            fake_pred2 = discriminator(fake_img2)
            fake_pred3 = discriminator(fake_img3)
            fake_pred4 = discriminator(fake_img4)


            g_loss1 = g_nonsaturating_loss(fake_pred)
            g_loss2 = g_nonsaturating_loss(fake_pred1)
            g_loss3 = g_nonsaturating_loss(fake_pred2)
            g_loss4 = g_nonsaturating_loss(fake_pred3)
            g_loss5 = g_nonsaturating_loss(fake_pred4)


            g_loss = 1/5*(g_loss1+g_loss2+g_loss3+g_loss4+g_loss5)
        
        else:
            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

        
        g_regularize = i % args.g_reg_every == 0
        loss_dict["g"] = g_loss

        generator.zero_grad()
        if g_regularize:
        	g_loss.backward(retain_graph=True)
        else:
        	g_loss.backward()
        g_optim.step()


        if g_regularize:
            if args.rtil==True:
                path_batch_size=args.path_batch_shrink
                noise1 = mixing_noise(
                    path_batch_size, args.latent, args.mixing, device)
                noise2 = mixing_noise(
                    path_batch_size, args.latent, args.mixing, device)
                noise3 = mixing_noise(
                    path_batch_size, args.latent, args.mixing, device)
                noise4 = mixing_noise(
                    path_batch_size, args.latent, args.mixing, device)

                
            
                fake_img1,p_l1 = generator(noise1,rtil=0, path_reg=True)
                fake_img2,p_l2 = generator(noise2,rtil=1, path_reg=True)
                fake_img3,p_l3 = generator(noise3,rtil=2, path_reg=True)
                fake_img4,p_l4 = generator(noise4,rtil=3, path_reg=True)
                fake_img5,p_l5 = generator(noise4,rtil=4, path_reg=True)
            


                path_loss1, mean_path_length1, path_lengths1 = g_path_regularize(p_l1,mean_path_length1)
                path_loss2, mean_path_length2, path_lengths2 = g_path_regularize(p_l2,mean_path_length2)
                path_loss3, mean_path_length3, path_lengths3 = g_path_regularize(p_l3,mean_path_length3)
                path_loss4, mean_path_length4, path_lengths4 = g_path_regularize(p_l4,mean_path_length4)
                path_loss5, mean_path_length5, path_lengths5 = g_path_regularize(p_l5,mean_path_length5)
          
          

                path_loss=(1/5)*(path_loss1+path_loss2+path_loss3+path_loss4+path_loss5)

                mean_path_length=(1/5)*(mean_path_length1+ mean_path_length2 + mean_path_length3 + mean_path_length4 + mean_path_length5)

                path_lengths=(1/5)*(path_lengths1+ path_lengths2 + path_lengths3 + path_lengths4 + path_lengths5)

                generator.zero_grad()
                weighted_path_loss1 = args.path_regularize * args.g_reg_every * path_loss1
                weighted_path_loss2 = args.path_regularize * args.g_reg_every * path_loss2
                weighted_path_loss3 = args.path_regularize * args.g_reg_every * path_loss3
                weighted_path_loss4 = args.path_regularize * args.g_reg_every * path_loss4
                weighted_path_loss5 = args.path_regularize * args.g_reg_every * path_loss5
               

                
                if args.path_batch_shrink:
                    weighted_path_loss1 += 0 * fake_img1[0, 0, 0, 0]
                    weighted_path_loss2 += 0 * fake_img2[0, 0, 0, 0]
                    weighted_path_loss3 += 0 * fake_img3[0, 0, 0, 0]
                    weighted_path_loss4 += 0 * fake_img4[0, 0, 0, 0]
                    weighted_path_loss5 += 0 * fake_img5[0, 0, 0, 0]
                    
                    weighted_path_loss= 1/5* (weighted_path_loss1 + weighted_path_loss2  + weighted_path_loss3 +  weighted_path_loss4 + weighted_path_loss5)
                
            else:
                path_batch_size=args.path_batch_shrink

                noise1 = mixing_noise(path_batch_size, args.latent, args.mixing, device)

                fake_img1,p_l1 = generator(noise1,tail=0, path_reg=True)

                path_loss1, mean_path_length1, path_lengths1 = g_path_regularize(p_l1,mean_path_length1)

                path_loss=path_loss1

                mean_path_length=mean_path_length1

                path_lengths=path_lengths1

                generator.zero_grad()
                weighted_path_loss1 = args.path_regularize * args.g_reg_every * path_loss1
                
                if args.path_batch_shrink:
                    weighted_path_loss1 += 0 * fake_img1[0, 0, 0, 0]

                    weighted_path_loss=weighted_path_loss1
                
          



            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )



            if i % 1000 == 0:
                if args.rtil==True:
                    with torch.no_grad():
                        g_ema.eval()
                        sample, _, = g_ema([sample_z], rtil=0)
                        sample1, _, = g_ema([sample_z1], rtil=1)
                        sample2, _, = g_ema([sample_z2], rtil=2)
                        sample3, _, = g_ema([sample_z3], rtil=3)
                        sample4, _, = g_ema([sample_z4], rtil=4)


                        utils.save_image(
                            sample,
                            f"rtil/sample/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            sample1,
                            f"rtil/sample1/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            sample2,
                            f"rtil/sample2/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            sample3,
                            f"rtil/sample3/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            sample4,
                            f"rtil/sample4/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                else:
                    with torch.no_grad():
                        g_ema.eval()
                        sample, _, = g_ema([sample_z], rtil=0)
                        utils.save_image(sample,
                            f"van/sample/{str(i).zfill(6)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),)
                    

            if i % 5000 == 0:
                if args.rtil==True:
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p,
                        },
                        f"rtil/checkpoint/{str(i).zfill(6)}.pt",
                    )
                    np.save(f"rtil/loss_stat/{str(i).zfill(6)}.npy", loss_dict)
                else:
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p,
                        },
                        f"van/checkpoint/{str(i).zfill(6)}.pt",
                    )
                    np.save(f"van/loss_stat/{str(i).zfill(6)}.npy", loss_dict)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    
    parser.add_argument("--rtil",type=int,default=True,help="Type of Training Mehotd ",)

    parser.add_argument(
        "--path", type=str, default= './ffhq', help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2',
                        help='model architectures (stylegan2)')

    parser.add_argument('--multi_gpu', type=bool, default=True,
                        help='Multi GPU Training')
    parser.add_argument(
        "--iter", type=int, default=600000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=6, help="batch sizes for each gpus"
    )
   
    parser.add_argument(
        "--n_sample",
        type=int,
        default=15,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )

    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002,
                        help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", type=bool, default=False, help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    parser.add_argument(
        "--datafolder",
        type=int,
        default=True,
        help="Import Data from Folder",
    )
    

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator


    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.multi_gpu:
        generator = nn.DataParallel(
            generator,
            device_ids=[0,1]
        )

        g_ema = nn.DataParallel(
            g_ema,
            device_ids=[0,1],
        )


        discriminator = nn.DataParallel(
            discriminator,
            device_ids=[0,1])

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

   

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if args.datafolder:

        dataset = dset.ImageFolder(root=args.path, transform=transform)
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True,
                                 distributed=args.distributed),
        )

    else:

        dataset = MultiResolutionDataset(args.path, transform, args.size)
    


    train(args, loader, generator, discriminator,
          g_optim, d_optim, g_ema, device)
