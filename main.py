from fastai.vision.all import *
from fastai.metrics import *

path = Path('ProjData/insect/')

def is_butterfly(x):
    return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_butterfly, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
if __name__ == "__main__":
    learn.fine_tune(2)