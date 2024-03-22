import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
from torch.cuda.amp import autocast

def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch, device, beta, print_freq):
    
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses_c = AverageMeter()
    losses_a = AverageMeter()
    losses_d = AverageMeter()

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        lr_imgs = convert_image(lr_imgs, source='[0, 1]', target='imagenet-norm', device=device)
        hr_imgs = convert_image(hr_imgs, source='[0, 1]', target='imagenet-norm', device=device)

        # GENERATOR UPDATE
        sr_imgs = generator(lr_imgs)
        sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm', device=device)
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

        sr_discriminated = discriminator(sr_imgs)

        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        optimizer_g.zero_grad()
        perceptual_loss.backward()
        optimizer_g.step()

        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        optimizer_d.step()

        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

def train_pre(train_loader, generator, discriminator, truncated_vgg19, pre_train_loss_criterion, content_loss_criterion, adversarial_loss_criterion,
          optimizer_p, optimizer_g, optimizer_d, epoch, device, beta, print_freq, max_epoch):
    
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses_c = AverageMeter()
    losses_a = AverageMeter()
    losses_d = AverageMeter()
    losses_p = AverageMeter()

    start = time.time()

    

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # 前一半的epoch作预训练
        if epoch < max_epoch/2 :
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            lr_imgs = convert_image(lr_imgs, source='[0, 1]', target='imagenet-norm', device=device)
            hr_imgs = convert_image(hr_imgs, source='[0, 1]', target='imagenet-norm', device=device)

            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm', device=device)
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            pre_content_loss = pre_train_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            optimizer_p.zero_grad()
            pre_content_loss.backward()
            optimizer_p.step()

            losses_p.update(pre_content_loss.item(), lr_imgs.size(0))

            batch_time.update(time.time() - start)
            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]----'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                    'Cont. Loss {loss_p.val:.4f} ({loss_p.avg:.4f})'.format(epoch,
                                                                            i,
                                                                            len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time,
                                                                            loss_p=losses_p))
        else:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            lr_imgs = convert_image(lr_imgs, source='[0, 1]', target='imagenet-norm', device=device)
            hr_imgs = convert_image(hr_imgs, source='[0, 1]', target='imagenet-norm', device=device)

            # GENERATOR UPDATE
            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm', device=device)
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            perceptual_loss = content_loss + beta * adversarial_loss

            optimizer_g.zero_grad()
            perceptual_loss.backward()
            optimizer_g.step()

            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # DISCRIMINATOR UPDATE
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
            
            optimizer_d.zero_grad()
            adversarial_loss.backward()
            optimizer_d.step()

            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            batch_time.update(time.time() - start)
            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]----'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                    'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                    'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                    'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                            i,
                                                                            len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time,
                                                                            loss_c=losses_c,
                                                                            loss_a=losses_a,
                                                                            loss_d=losses_d))
    
    # 前一半的epoch作预训练
    if epoch < max_epoch/2 :
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space
    else:
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated


def train_half(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch, device, beta, print_freq):
    
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses_c = AverageMeter()
    losses_a = AverageMeter()
    losses_d = AverageMeter()

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        with autocast(): # 半精
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            lr_imgs = convert_image(lr_imgs, source='[0, 1]', target='imagenet-norm', device=device)
            hr_imgs = convert_image(hr_imgs, source='[0, 1]', target='imagenet-norm', device=device)

            # GENERATOR UPDATE
            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm', device=device)
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            sr_discriminated = discriminator(sr_imgs)

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            perceptual_loss = content_loss + beta * adversarial_loss

            optimizer_g.zero_grad()
            perceptual_loss.backward()
            optimizer_g.step()

            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # DISCRIMINATOR UPDATE
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            optimizer_d.step()

            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            batch_time.update(time.time() - start)
            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]----'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                    'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                    'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                    'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                            i,
                                                                            len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time,
                                                                            loss_c=losses_c,
                                                                            loss_a=losses_a,
                                                                            loss_d=losses_d))

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

