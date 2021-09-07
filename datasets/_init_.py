
from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd, nihdataset, tooth, FlightDataset, emdataset
from torch.utils.data import DataLoader
from natsort import natsorted
from glob import glob
from torchvision import transforms
import torch

def make_data_loader(args, **kwargs):
    if args.dataset == 'Nih_dataset':
        dataset_base_path = '/your_own_path/datasets/nih_pancras/'
        target_path = natsorted(glob(dataset_base_path + '/mask/*.png'))
        image_paths = natsorted(glob(dataset_base_path + '/img/*.png'))
        # border_paths = natsorted(glob(dataset_base_path + '/border/*.png'))

        target_val_path = natsorted(glob(dataset_base_path + '/val_mask/*.png'))
        image_val_path = natsorted(glob(dataset_base_path + '/val_img/*.png'))
        border_val_paths = natsorted(glob(dataset_base_path + '/val_border/*.png'))
        nihdataset_train = nihdataset.Nih_dataset(image_paths=image_paths,
                                                  target_paths=target_path)
        nihdataset_val = nihdataset.Nih_dataset(image_paths=image_val_path,
                                                target_paths=target_val_path)

        train_loader = DataLoader(nihdataset_train, batch_size=1, shuffle=True, num_workers=4)
        val_loader = DataLoader(nihdataset_val, batch_size=1, shuffle=True, num_workers=4)
        test_loader = None
        return train_loader, val_loader
    elif args.dataset == 'EM_dataset':
        dataset_base_path = '/your_own_path/datasets/em_challenge/'
        target_path = natsorted(glob(dataset_base_path + '/mask/*.PNG'))
        image_paths = natsorted(glob(dataset_base_path + '/data/*.PNG'))
        # border_paths = natsorted(glob(dataset_base_path + '/border/*.png'))

        target_val_path = natsorted(glob(dataset_base_path + '/val_mask/*.PNG'))
        image_val_path = natsorted(glob(dataset_base_path + '/val_img/*.PNG'))
        # border_val_paths = natsorted(glob(dataset_base_path + '/val_border/*.png'))
        emdataset_train = emdataset.EM_dataset(image_paths=image_paths,
                                                  target_paths=target_path)
        emdataset_val = emdataset.EM_dataset(image_paths=image_val_path,
                                                target_paths=target_val_path)

        train_loader = DataLoader(emdataset_train, batch_size=2, shuffle=True, num_workers=4)
        val_loader = DataLoader(emdataset_val, batch_size=1, shuffle=True, num_workers=4)
        test_loader = None
        return train_loader, val_loader, test_loader

    elif args.dataset == 'Tooth':
      
    else:
        raise NotImplementedError
