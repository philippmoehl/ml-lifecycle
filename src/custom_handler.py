import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import map_class_to_label



class LeafClassifier(ImageClassifier):
    """
    LeafClassifier handler class.
    """

    image_processing = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

    ])

    topk = 5

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):
        probs = data.softmax(axis=1)
        confs, preds = torch.topk(probs, self.topk, dim=1)
        confs = confs.tolist()
        preds = preds.tolist()
        return map_class_to_label(confs, self.mapping, preds)
