

class CompetitionHandler : 



    def __init__(self,cfg) : 


        self.__import()
        self.cfg=cfg
    

    def __import(self) :



        import numpy as np 
        import os
        import radiomics
        import pandas as pd 


    def __InitializeDataHandler(self) : 
         
        
        self.DataHandler=DataHandler(self.cfg["DataHandler_cfg"])


    

    def __submit(self) : 
         
         df=self.DataHandler.get_data("test")

         for index,row in df.iterrows(): 
              
              row["target"]=5

         df.to_csv(self.cfg["submission_path"]) 

    

    def __call__(self) : 



        self.__submit()
    





class DataHandler : 


    def __init__(self,dic_path : dict) : 


            self.__check_dic_data()
    


    def __check_dic_data(self,dic) : 
          

          """
          This function to make sure the dictionnary is correctly formated
          
          :param self: Description

          """

          assert "BASE_DIR" in list(dic.keys())
          assert "train_path" in list(dic.keys())
          assert "test_path" in list(dic.keys())
      

          self.dic_path={k : os.path.join(dic["BASE_DIR"],i) for k,i in dic.items if k!="BASE_dir"} | {"BASE_DIR" : dic["BASE_DIR"]}



          
    def get_data(self,type_:str) -> pd.DataFrame:  
         

         assert type_  in ["train","test"]

         if type=="train" : 
              
              return pd.read_csv(self.dic_path["train_path"])
        
         elif type=="test" : 
              
              return pd.read_csv(self.dic_path["test_path"])
         


         




         
         
         
         

