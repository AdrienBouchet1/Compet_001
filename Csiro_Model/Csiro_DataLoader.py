from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np 




class CsiroDataSet(Dataset) : 


    """
    Attention : Ce dataset n'est destiné qu'à l'utilisatio

    """

    def __init__(self,CSV_path,full : str = True,device : stre ="cuda") : 

        """
        Full if we have all tabular data (for training) 
        """
        
        self.device=device
        self.full=full
        self.csv=pd.read_csv(CSV_path)
        

        ###instanciation des transforms
        transform_list = [
            transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        ]
        self.transform_norm = transforms.Compose(transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        self.list_path=self.csv["image_path"].unique().tolist()
    
    def __transform(self, x):

        
        return self.transform_norm(x)

    def __augmentation(self, x): 

        pass

        
    def __len__(self): 

        
        return len(self.list_path)

    
    def __getitem__(self,idx): 

        path=self.list_path[idx]
        if self.full : 
            
            sub_data=self.csv[self.csv["image_path"]==path]

        else : 
            sub_data=self.csv[self.csv["image_path"]==path][["sample_id","image_path","target_name","target"]]
            
    
        img = Image.open(self.list_path[idx])
        img = self.__transform(img).to("cuda") 




        ###Mise en forme de l'output
        cols_for_x = [col for col in sub_data.columns if col not in ['target_name', 'target',"sample_id","image_path"]]
        
        dic_ = {
        
    "x": {col: sub_data.iloc[0][col] for col in cols_for_x} ,
    "y": {row['target_name']: row['target'] for _, row in sub_data.iterrows()}
        }

        dic_["x"]["image"]=img
        
        return dic_


        







        

        