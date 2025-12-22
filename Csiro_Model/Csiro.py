import torch.nn as nn


class Csiro(nn.Module) : 


    def __init__(self, Hiera_instance): 
        """
        Hiera_instance doit être une instance de modèle Hiera (on laisse la souplesse de choisir quel type d'instance, donc on le saisi en tant qu'argument
        """
      
        super().__init__() 

        self.device="cuda"
        self.__build_model(Hiera_instance)


    def __build_model(self,Hiera_instance):
        
        ### Ici, on considère que les poids d'intérêt pour HIERA ont déja été chargés à l'initialisation de l'instance
        self.hiera=Hiera_instance.to(self.device)
        
        
    def forward(self,x) : 

        """
        Dans le cas classique (sans meta données) x (donc l'image) doit être un tenseur normalisé de taille (1,3,H,W)

        """
        
        _,x2=self.hiera(x,return_intermediates=True) 
        
        
        return x2
        
        

         



        