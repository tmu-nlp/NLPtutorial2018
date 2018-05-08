
class SimpleBlender:
    def __init__(self, unk_rate=0.05):
        self.unk = unk_rate

    def unk_rate(self, *words):
        return self.unk

class WittenBell:

    def unk_rate(self, *words):
        pass