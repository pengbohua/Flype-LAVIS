import json
import os
from torch.utils.data import Dataset
from PIL import Image
from lavis.models import load_model_and_preprocess


class Img_Txt_Dataset(Dataset):
    def __init__(self, img_pth, txt_pth, model, preprocess):
        super(Img_Txt_Dataset, self).__init__()
        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.images = os.listdir(img_pth)
        self.texts = json.load(open(txt_pth, 'r'))
        self.preprocess = preprocess
        self.model = model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        images = self.preprocess(Image.open(os.path.join(self.img_pth, self.images[idx]))).to(device)

        tmp_txt = self.texts[image_name]['txt'] if len(self.texts[image_name]['txt']) <= 77 else self.texts[image_name][
                                                                                                     'txt'][:77]
        texts = clip.tokenize(tmp_txt).to(device)
        labels = clip.tokenize(misogyny_label if self.texts[image_name]['misogynous'] == '1' else no_misogyny_label).to(
                device)

        labels_lis = self.texts[image_name]['misogynous']
        return images, texts, labels, labels_lis
