import torch.nn as nn


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
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 5) )
            
    def forward(self,x) : 

        """
        Dans le cas classique (sans meta données) x (donc l'image) doit être un tenseur normalisé de taille (1,3,H,W)

        """
        
        _,x2=self.hiera(x,return_intermediates=True) 

        
        pooled=x2[-1].mean(dim=(1, 2))
        print("pooled dim : ", pooled.shape)
        y=self.mlp(pooled)
        
        return y
        
        

         



        
