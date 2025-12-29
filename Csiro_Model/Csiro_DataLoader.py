from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np 
from albumentations.pytorch import ToTensorV2


import albumentations as A

class CsiroDataSet(Dataset) : 


    """
    Attention : Ce dataset n'est destiné qu'à l'utilisatio

    """

    def __init__(self,CSV_path,full : str = True,device:str="cuda", augment: bool = True, resize:bool=True) : 

        """
        Full if we have all tabular data (for training) 
        """
        
        self.device=device
        self.full=full
        self.csv=pd.read_csv(CSV_path)
        self.augment = augment

        ###instanciation des transforms
        transform_list = [
            transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        ]

        if resize : 
            self.transform_norm = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])

            
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(1,3), p=0.3),  # flou faible
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.Resize(224,224),  # toujours resize à la fin pour être sûr
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])

        else:  
            
            self.transform_norm = A.Compose([
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])

            
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
               
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(1,3), p=0.3),  # flou faible
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])

        

        self.list_path=self.csv["image_path"].unique().tolist()

        
    
    def __transform(self, x):

        
        return self.transform_norm(x)
    
    def __len__(self): 

        
        return len(self.list_path)

    
    def __getitem__(self,idx): 

        path=self.list_path[idx]
        if self.full : 
            
            sub_data=self.csv[self.csv["image_path"]==path]

        else : 
            sub_data=self.csv[self.csv["image_path"]==path][["sample_id","image_path","target_name","target"]]
            
        img = np.array(Image.open(path).convert("RGB"))
        assert img.shape==(1000,2000,3), "problème : shape {}".format(img.shape)
        if self.augment:
            img = self.augmentations(image=img)["image"]
        else:
            img = self.transform_norm(image=img)["image"]

        #  img = self.__transform(img).to(self.device) 

        img=img.to(self.device)


        ###Mise en forme de l'output
        cols_for_x = [col for col in sub_data.columns if col not in ['target_name', 'target',"sample_id","image_path"]]
        
        dic_ = {
        
    "x": {col: sub_data.iloc[0][col] for col in cols_for_x} ,
    "y": {row['target_name']: row['target'] for _, row in sub_data.iterrows()}
        }

        dic_["x"]["image"]=img
        
        return dic_


        







        

        