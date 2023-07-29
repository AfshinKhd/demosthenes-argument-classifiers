import pandas as pd

class CfgMaper(dict):

    def __init__(self, *args, **kwargs):
        super(CfgMaper, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(CfgMaper, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(CfgMaper, self).__delitem__(key)
        del self.__dict__[key]

    def merge_file(self, new_conf) :
        merged = CfgMaper({**self.__dict__, **new_conf})
        # Convet Model in yaml script to CfgMaper 
        merged.MODEL = CfgMaper(merged.MODEL)
        return merged
       




def preprocessing_data(df, label_name):
    # remove nan(float) elements from dataset
    df  = df.dropna(subset=[label_name]).reset_index(drop=True)

    # get back str type of label as list
    df[label_name] = df[label_name].apply(lambda x: [x] if isinstance(x, str) else x)
    
    # split the lists into separate elements and create new rows
    df = df.apply(lambda x: x.explode() if x.name == label_name else x).reset_index(drop=True)

    return df

def prepare_AD_label(label):
   if label == 'prem':
     return [1,0,0]
   elif label == 'conc':
       return [0,1,0] 
   else: 
       return [0,0, 1]
   
def prepare_AC_label(label):
   if label == 'prem':
     return [1,0]
   # label == conc
   else :
       return [0,1] 

   
def prepare_TC_label(label):
   if label == 'L':
     return [1,0]
   # label == F
   else:
       return [0,1] 

# def prepare_SC_label(label):
#     res = [0,0,0,0,0,0]
#     print("type: ", type(label))
#     for l in label:
#         res = _SC_label(res, l)
#     return res
   
def prepare_SC_label(l):
   res = [0,0,0,0,0,0]
   if l == 'Rule':
       res[0] = 1
       return res
   elif l == 'Itpr':
       res[1] = 1
       return res
   elif l == 'Prec':
      res[2] = 1
      return res
   elif l == 'Class':
      res[3] = 1
      return res
   elif l == 'Princ':
       res[4] = 1
       return res
   # label == Aut
   else: 
       res[5] = 1
       return res

