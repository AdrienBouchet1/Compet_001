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
                    #nn.Linear(1152, 768),
                    #nn.ReLU(),
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256, 5))

    def __tile_batch(self, x, tile_size=224, stride=224):
        """
        x: (B,3,H,W)
        retourne:
            tiles: (N_tiles,3,224,224)
            tile_to_img: (N_tiles,)
        """
        B, C, H, W = x.shape
        tiles = []
        tile_to_img = []
    
        for b in range(B):
            for y in range(0, H - tile_size + 1, stride):
                for x_ in range(0, W - tile_size + 1, stride):
                    tile = x[b, :, y:y+tile_size, x_:x_+tile_size]
                    tiles.append(tile)
                    tile_to_img.append(b)
    
        tiles = torch.stack(tiles)
        tile_to_img = torch.tensor(tile_to_img, device=x.device)
    
        return tiles, tile_to_img


    
    def __tile_one_image(self, x, tile_size=224, stride=224):
        """
        x: (3,H,W)
        retourne: (N_tiles,3,224,224)
        """
        C, H, W = x.shape
        tiles = []
    
        for y in range(0, H - tile_size + 1, stride):
            for x_ in range(0, W - tile_size + 1, stride):
                tiles.append(x[:, y:y+tile_size, x_:x_+tile_size])
    
        return torch.stack(tiles)

        
    def forward(self, x):

            # ======================
            # Cas image 224x224
            # ======================
            if x.shape[-2:] == (224, 224):
                _, x2 = self.hiera(x, return_intermediates=True)
                feat_deep = x2[-1].mean(dim=(1, 2))  # (B,768)
                return self.mlp(feat_deep)
        
            # ======================
            # Cas image grande (batch)
            # ======================
            elif x.shape[-2:] == (500,1000):
        
                B = x.shape[0]
                feats_batch = []
        
                for b in range(B):
             
                    # ---- découpe image b ----
                    tiles = self.__tile_one_image(x[b])
                    tiles = tiles.to(self.device)
                    
                  
                    # ---- passage dans Hiera ----
                    _, x2 = self.hiera(tiles, return_intermediates=True)
                    
                    
                    # ---- embedding par tuile ----
                    feat_tiles = x2[-1].mean(dim=(1, 2))  # (N_tiles,768)
            
                    
        
                    # ---- moyenne POUR CETTE IMAGE ----
                    feat_img = feat_tiles.mean(dim=0)     # (768,)
                    feats_batch.append(feat_img)
                    
        
                feats_batch = torch.stack(feats_batch)     # (B,768)
               
                # ---- MLP final ----
                y = self.mlp(feats_batch)
                return y
        
            else:
                raise Exception(f"problème de shape : {x.shape[-2:]}")
                        
                

         



        
