
import argparse
import os
import numpy as np
import torch
import cv2


from tqdm import tqdm
from torchvision import transforms
from modeling.UNet import UNet
from modeling.UNet_Light import Deep_Wise_UNet
from modeling.UNet_Light import ThinUNet
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from thop import profile
from thop import clever_format

red       = [60,20,220]
green     = [144,238,144]
blue      = [255,144,30]
yellow    = [255,255,0]
orange    = [0,165,255]
white     = [255, 255, 255]
black     = [0 ,   0, 0 ]
burgundy  = [7 , 0  ,30]

def create_mask(img, mask_val, back_color):
    new_img = np.zeros(img.shape)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if np.abs(pixel - back_color) > 10:
                new_img[i][j] = mask_val
            else:
                new_img[i][j] = back_color
    new_img = 255 - new_img
    return new_img

def combine(img1, img2, img_color1, img_color2, union_color, orig_img, bkgnd_color):
    out_img = np.zeros(img1.shape)
    out_img = to_rgb(out_img)
    for irow, rows in enumerate(img1):
        for icol, cols in enumerate(rows):
            # import ipdb as pdb; pdb.set_trace()
            img1_cond = not (img1[irow][icol] == bkgnd_color).all()
            img2_cond = not (img2[irow][icol] == bkgnd_color).all()
            if img1_cond or img2_cond:
                if img1_cond and img2_cond:
                    out_img[irow][icol] = union_color
                elif img1_cond:
                    out_img[irow][icol] = img_color1
                else:
                    out_img[irow][icol] = img_color2
            else:
                out_img[irow][icol] = orig_img[irow][icol]
    return out_img


def to_rgb(gray_img):
    # import ipdb as pdb; pdb.set_trace()
    img_size = gray_img.shape
    rgb_img = np.zeros((img_size[0], img_size[1], 3))
    rgb_img[:, :, 0] = gray_img
    rgb_img[:, :, 1] = gray_img
    rgb_img[:, :, 2] = gray_img
    return rgb_img
class Tester(object):
    def __init__(self, args):
        self.args = args
        self.nclass = 1
        # Define Saver
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        model = UNet(1, 1, 64, sync_bn=False)

        input = torch.randn(1, 1, 320, 640)
        flops, params = profile(model, inputs=(input,))
        flops, params = clever_format([flops, params], "%.3f")
        print('flops: ', flops, 'params: ', params)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0



    def test(self, epoch):
        model_name = 'Unet'
        load_path = '/path/'+model_name+'.pkl'
        self.model.load_state_dict(torch.load(load_path), strict=False)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader)
        test_loss = 0.0
        dice = 0
        data_len = len(self.val_loader)
        for i, batch in enumerate(tbar):
            image, target, img_name = batch
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output

            toimage = transforms.ToPILImage()
            output_write = toimage(pred.cpu().squeeze(0)).resize((200, 200))
            output_write = np.array(output_write)
            output_write[output_write > 100] = 255

            toimage = transforms.ToPILImage()
            mask_write = toimage(target.cpu().squeeze(0)).resize((200, 200))
            mask_write = np.array(mask_write)
            each_dice = self.evaluator.dice_loss(pred, target)
 
            dice += each_dice

            gt_img = target.cpu().numpy()[0][0]
            in_img = image.cpu().numpy()[0][0]
            pred_img = output.cpu().numpy()[0][0]

            gt_img *= 255
            pred_img *= 255
            in_img *= 255
            gt_img = create_mask(gt_img, 0, 255)
            pred_img = create_mask(pred_img, 0, 255)
            # preds = pred.detach().cpu().numpy()[0][0]
            pred_over_layed = combine(gt_img, pred_img, red, orange, blue, in_img, black)

          

            cv2.imwrite("/path/"+model_name+"/" + img_name[0] +"_"+model_name+ "_dice:(" + str(
                round(each_dice.item(), 4)) + ").png", pred_over_layed)
      
            cv2.imwrite("/path"+model_name+"/" + img_name[0] + "_mask.png", mask_write)


        print('dice:', float('%.4f' % (dice.item() / data_len)))
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        return dice.item() / data_len


def main():
    parser = argparse.ArgumentParser(description="PyTorch Testing EM_dataset")
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='EM_dataset',
                        choices=['Nih_dataset', 'Tooth','EM_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--freeze-bn', action='store_true', default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='calc',
                        choices=['ce', 'focal ', 'calc'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {

            'nih_dataset': 1,
            'em_dataset':1,
            'tooth': 1
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 6 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
     
            'nih_dataset': 0.01,
            'em_dataset': 0.03,
            'tooth': 0.035
        }
        args.lr = lrs[args.dataset.lower()] / (6 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    print('Starting Epoch:', tester.args.start_epoch)
    print('Total Epoches:', tester.args.epochs)
    best_dice = 0
    for epoch in range(tester.args.start_epoch, tester.args.epochs):
        dice = tester.test(epoch)
        if dice > best_dice:
            best_dice = dice
    print('Finish training!')
    print('best_dice:', float('%.3f' % best_dice))


if __name__ == "__main__":
    main()
