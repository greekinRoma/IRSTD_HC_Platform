from .TopHat import *
from .MPCM import *
from .WSLCM import *
from .IPI import *
class Algorithms(nn.Module):
    def __init__(self):
        super().__init__()
        self.alg_dict ={
            'TopHat':TopHat(),
            'MPCM':MPCM(),
            'WSLCM':WSLCM(),
            'IPI':IPI()
        }
        self.model = None
    def detect(self,name):
        if name not in self.alg_dict.keys():
            return False
        else:
            return True
    def set_algorithm(self,name):
        self.model=self.alg_dict[name]
    def forward(self,inp):
        return self.model(inp)