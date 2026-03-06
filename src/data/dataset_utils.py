import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode

from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img


class CDD11(Dataset):
    def __init__(self, args, split: str = "train", subset: str = "all"):
        super(CDD11, self).__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.dataset_split = split
        self.subset = subset
        if split == "train":
            self.patch_size = args.patch_size
        else:
            self.patch_size = 64

        self._init()

    def __getitem__(self, index):
        # Randomly select a degradation type
        if self.dataset_split == "train":
            degradation_type = random.choice(list(self.degraded_dict.keys()))
            degraded_image_path = random.choice(self.degraded_dict[degradation_type])
        else:
            degradation_type = self.subset
            degraded_image_path = self.degraded_dict[degradation_type][index]
        
        # Select a degraded image within that type

        degraded_name = os.path.basename(degraded_image_path)

        # Get the corresponding clean image based on the file name
        image_name = os.path.basename(degraded_image_path)
        assert degraded_name == image_name
        clean_image_path = os.path.join(os.path.dirname(self.clean[0]), image_name)

        # Load the images
        lr = np.array(Image.open(degraded_image_path).convert('RGB'))
        hr = np.array(Image.open(clean_image_path).convert('RGB'))
        # Apply random augmentation and crop
        if self.dataset_split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # Convert to tensors
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [clean_image_path, degradation_type], lr, hr

    def __len__(self):
        return sum(len(images) for images in self.degraded_dict.values())

    def _init(self):
        # data_dir = os.path.join(self.args.data_file_dir, "cdd11")
        data_dir = self.args.data_file_dir
        self.clean = sorted(glob.glob(os.path.join(data_dir, f"{self.dataset_split}/clear", "*.png")))

        if len(self.clean) == 0:
            raise ValueError(f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        self.degraded_dict = {}
        allowed_degradation_folders = self._filter_degradation_folders(data_dir)
        for folder in allowed_degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            degraded_images = sorted(glob.glob(os.path.join(folder, "*.png")))
            
            if len(degraded_images) == 0:
                raise ValueError(f"No images found in {folder_name}")
            
            # scale dataset length
            if self.dataset_split == "train":
                degraded_images *= 2
            
            self.degraded_dict[folder_name] = degraded_images

    def _filter_degradation_folders(self, data_dir):
        degradation_folders = sorted(glob.glob(os.path.join(data_dir, self.dataset_split, "*/")))
        filtered_folders = [] 

        for folder in degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            if folder_name == "clear":
                continue

            degradation_count = folder_name.count('_') + 1

            if self.subset == "single" and degradation_count == 1:
                filtered_folders.append(folder)
            elif self.subset == "double" and degradation_count == 2:
                filtered_folders.append(folder)
            elif self.subset == "triple" and degradation_count == 3:
                filtered_folders.append(folder)
            elif self.subset == "all":
                filtered_folders.append(folder)

            elif self.subset not in ["single", "double", "triple", "all"]:
                if folder_name == self.subset:
                    filtered_folders.append(folder)

        print(f"Degradation type mode: {self.subset}")
        print(f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders

    def _crop_patch(self, img_1, img_2):

        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
            
class AIOTrainDataset(Dataset):
    def __init__(self, args):
        super(AIOTrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}
        
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        self._init_lr()
        self._merge_tasks()
            
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]
        
        if deg_type == "denoise_15" or deg_type == "denoise_25" or deg_type == "denoise_50":
            
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = self.crop_transform(hr)
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, de_id)
        else:
            if deg_type == "dehaze":
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
                
            else:
                hr_sample = self.hr[idx]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
        
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        
        return [lr_sample["img"], de_id], lr, hr
        
    
    def __len__(self):
        return len(self.lr)
    
    
    def _init_lr(self):
        # synthetic datasets
        if 'synllie' in self.de_type:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'deblur' in self.de_type:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'derain' in self.de_type:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.de_type:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_25' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_50' in self.de_type:
            self._init_clean(id=0)
            
    def _merge_tasks(self):
        self.lr = []
        self.hr = []
        # synthetic datasets
        if "synllie" in self.de_type:
            self.lr += self.synllie_lr
            self.hr += self.synllie_hr
        if "denoise_15" in self.de_type:
            self.lr += self.s15_ids
            self.hr += self.s15_ids
        if "denoise_25" in self.de_type:
            self.lr += self.s25_ids
            self.hr += self.s25_ids
        if "denoise_50" in self.de_type:
            self.lr += self.s50_ids
            self.hr += self.s50_ids
        if "deblur" in self.de_type:
            self.lr += self.deblur_lr 
            self.hr += self.deblur_hr
        if "derain" in self.de_type:
            self.lr += self.derain_lr 
            self.hr += self.derain_hr
        if "dehaze" in self.de_type:
            self.lr += self.dehaze_lr 
            self.hr += self.dehaze_hr

        print(len(self.lr))
   
            
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/Train/Low_light_Enh/low"
        targets = self.args.data_file_dir + "/Train/Low_light_Enh/gt"
        
        self.synllie_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.synllie_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.synllie_counter = 0
        print("Total SynLLIE training pairs : {}".format(len(self.synllie_lr)))
        self.synllie_lr = self.synllie_lr * 20
        self.synllie_hr = self.synllie_hr * 20
        print("Repeated Dataset length : {}".format(len(self.synllie_hr)))
    
    def _init_deblur(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.args.data_file_dir + "/Train/Deblur/blur"
        targets = self.args.data_file_dir + "/Train/Deblur/sharp"
        
        self.deblur_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.deblur_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.deblur_counter = 0
        print("Total Deblur training pairs : {}".format(len(self.deblur_hr)))
        self.deblur_lr = self.deblur_lr * 5
        self.deblur_hr = self.deblur_hr * 5
        print("Repeated Dataset length : {}".format(len(self.deblur_hr)))
        
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/Train/Derain/rainy"
        targets = self.args.data_file_dir + "/Train/Derain/gt"

        
        self.derain_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.derain_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.derain_counter = 0
        print("Total Derain training pairs : {}".format(len(self.derain_lr)))
        self.derain_lr = self.derain_lr * 120
        self.derain_hr = self.derain_hr * 120
        print("Repeated Dataset length : {}".format(len(self.derain_hr)))
        
    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/Train/Dehaze/synthetic/"
        targets = self.args.data_file_dir + "/Train/Dehaze/original"
        
        self.dehaze_lr = []
        for part in ["part1", "part2", "part3", "part4"]:
            self.dehaze_lr += [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + part + "/*.jpg"))]
        
        self.dehaze_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.jpg"))]
        
        self.dehaze_counter = 0
        print("Total Dehaze training pairs : {}".format(len(self.dehaze_lr)))
        self.dehaze_lr = self.dehaze_lr
        self.dehaze_hr = self.dehaze_hr
        print("Repeated Dataset length : {}".format(len(self.dehaze_lr)))
        
    def _init_clean(self, id):
        inputs = self.args.data_file_dir + "/Train/Denoise"
        
        clean_bmp = sorted(glob.glob(inputs + "/*.bmp"))
        clean_jpg = sorted(glob.glob(inputs + "/*.jpg"))
        clean = clean_bmp + clean_jpg

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"img": x, "de_type":self.de_dict['denoise_15']} for x in clean]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"img": x, "de_type":self.de_dict['denoise_25']} for x in clean]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"img": x, "de_type":self.de_dict['denoise_50']} for x in clean]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    
    def _get_nonhazy_name(self, hazy_name):
        dehaze_dir = os.path.dirname(os.path.dirname(os.path.dirname(hazy_name)))
        original_dir = os.path.join(dehaze_dir, "original")
        
        filename = os.path.basename(hazy_name)
        base_name = filename.split('_')[0]
        ext = os.path.splitext(filename)[1]
        
        nonhazy_path = os.path.join(original_dir, base_name + ext)
        return nonhazy_path
        
    
class IRBenchmarks(Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()
        
        self.args = args
        self.benchmarks = args.benchmarks
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        
        self.toTensor = ToTensor()
        
        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)
        
        self._init_lr()
        
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        
        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or\
           "denoise_60" in self.benchmarks or "denoise_75" in self.benchmarks or "denoise_100" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_noise(hr, sigma)
        else:
            hr_sample = self.hr[idx]
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr
    
    def __len__(self):
        return len(self.lr)
    
    def _init_lr(self):
        if 'lolv1' in self.benchmarks:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'gopro' in self.benchmarks:
            self._init_deblurring(id=self.de_dict['deblur'])
        if 'derain' in self.benchmarks:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.benchmarks:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_60' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_75' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_100' in self.benchmarks:
            self._init_denoise(id=0)

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/gt"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + '.png'
        return nonhazy_name
    
    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
    
    ####################################################################################################
    ## DEBLURRING DATASET
    def _init_deblurring(self,id):
        inputs = self.args.data_file_dir + "/Test/Deblur/gopro/blur"
        targets = self.args.data_file_dir + "/Test/Deblur/gopro/sharp"

        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total Deblur testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## LLIE DATASET        
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/Test/Low_light_Enh/lol/input"
        targets = self.args.data_file_dir + "/Test/Low_light_Enh/lol/target"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total LLIE testing pairs : {}".format(len(self.hr)))
            
    ####################################################################################################
    ## DERAINING DATASET
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/Test/Derain/Rain100L/rainy"
        targets = self.args.data_file_dir + "/Test/Derain/Rain100L/gt"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Derain testing pairs : {}".format(len(self.hr)))

    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/Test/Dehaze/outdoor/hazy"
        targets = self.args.data_file_dir + "/Dehaze/outdoor/gt"

        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        
        self.hr = []
        for sample in self.lr:
            hazy_name = sample["img"]
            clean_name = self._get_nonhazy_name(hazy_name)
            self.hr.append({"img" : clean_name, "de_type":id})
        print("Total Dehazing testing pairs : {}".format(len(self.hr)))

    def _init_denoise(self, id):
        inputs = self.args.data_file_dir + "/Test/Denoise/CBSD68"  #CBSD68
        # inputs = self.args.data_file_dir + "/Test/Denoise/Kodak24"  #Kodak24
        # inputs = self.args.data_file_dir + "/Test/Denoise/urban100"  #urban100
        
        clean = [x for x in sorted(glob.glob(inputs + "/*.png"))]
        
        self.lr = [{"img" : x, "de_type":id} for x in clean]
        self.hr = [{"img" : x, "de_type":id} for x in clean]
        print("Total Denoise testing pairs : {}".format(len(self.lr)))



class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'synllie': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'rainy/')
            self.ids += [self.args.derain_path + 'rainy/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'outdoor/hazy/')
            self.ids += [self.args.dehaze_path + 'outdoor/hazy/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'blur/')
            self.ids += [self.args.gopro_path + 'blur/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.enhance_path + 'input/')
            self.ids += [self.args.enhance_path + 'input/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            base_dir = self.args.derain_path  
            degraded_filename = os.path.basename(degraded_name)
            clean_filename = 'norain-' + degraded_filename.split('rain-')[-1]
            gt_name = os.path.join(base_dir, 'gt', clean_filename)
        elif self.task_idx == 1:
            gt_name = degraded_name.replace("outdoor/gt", "outdoor/hazy")
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("sharp", "blur")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("target", "input")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    