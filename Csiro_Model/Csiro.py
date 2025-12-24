import torch.nn as nn
import torch

class Csiro(nn.Module) : 


    """ Version la plus simple : on utilise les features intermédiaires 1 par 1""" 
    
    def __init__(self, Hiera_instance,device:str="cuda"): 
        """
        Hiera_instance doit être une instance de modèle Hiera (on laisse la souplesse de choisir quel type d'instance, donc on le saisi en tant qu'argument
        """
      
        super().__init__() 

        self.device=device
        self.__build_model(Hiera_instance)


    def __build_model(self,Hiera_instance):
        
        self.hiera=Hiera_instance.to(self.device)
        self.mlp = nn.Sequential(
                    nn.Linear(1152, 768),
                    nn.ReLU(),
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 5))
            
    def forward(self,x) : 

        """
        Dans le cas classique (sans meta données) x (donc l'image) doit être un tenseur normalisé de taille (1,3,H,W)

        """
            
        _, x2 = self.hiera(x, return_intermediates=True)
    
        # ======================
        # Deep features (last)
        # ======================
        feat_deep = x2[-1].mean(dim=(1, 2))  # (B, C_deep)
    
        # ======================
        # Intermediate features (second last)
        # ======================
        feat_mid = x2[-2].mean(dim=(1, 2))   # (B, C_mid)
    
        # ======================
        # Concatenation
        # ======================
        pooled = torch.cat([feat_mid, feat_deep], dim=1)
    
        print("feat_mid :", feat_mid.shape)
        print("feat_deep:", feat_deep.shape)
        print("concat   :", pooled.shape)
    
        y = self.mlp(pooled)
    
        return y
        
        

         



        
