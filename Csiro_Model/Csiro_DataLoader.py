from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD




class CsiroDataSet(Dataset) : 
    
    def __init__(self,list_path) : 


        self.list_path=list_path

        ###instanciation des transforms
        transform_list = [
            transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        ]
        self.transform_norm = transforms.Compose(transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    
        
    def __transform(self, x):
        
        return self.transform_norm(x)
        
    def __augmentation(self, x): 
        
        pass
        
    def __len__(self): 

        return len(self.list_path) 

    def __getitem__(self,idx): 
        
        img = Image.open(self.list_path[idx])
        img = self.transform_norm(img).to("cuda") 
        
        return img 

        

        