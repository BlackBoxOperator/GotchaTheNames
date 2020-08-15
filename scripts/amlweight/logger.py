from gensim.models.callbacks import CallbackAny2Vec
from tqdm import *

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        if self.epoch == 0:
            self.bar = tqdm(total=model.epochs)

    def on_epoch_end(self, model):
        self.bar.update(1)
        self.epoch += 1
        if self.epoch == model.epochs:
            self.bar.close()
            self.bar = None
