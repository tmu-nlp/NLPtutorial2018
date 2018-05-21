import os, sys
sys.path.append(os.path.pardir)


class PosModel:
    def __init__(self):
        self.Pt = None
        self.Pe = None
        self.lam = None
        self.unk_rate = None

    def train(self, data, vocab_size=10**6):
        
        

        self.unk_rate = 1 / vocab_size
    
