



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
      

          self.dic_path=dic


          
          

