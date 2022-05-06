import random
from typing import Callable, Tuple
import torchvision.transforms as transforms
from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import torch
from torch import nn, Tensor
from typing import Union
from copy import deepcopy
from itertools import chain
from typing import Dict, List

import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as f


from torchvision.transforms import ToTensor
from os import cpu_count
from torchvision.models import resnet34
from torch.utils.data import DataLoader
import joblib

from sklearn.metrics import confusion_matrix

from Customdataset import InBreastDataset

class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)
    
def default_augmentation(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    return nn.Sequential(
    tf.Resize(size=image_size),
    RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
    aug.RandomGrayscale(p=0.2),
    aug.RandomHorizontalFlip(),
    RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
    aug.RandomResizedCrop(size=image_size),
    aug.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )
def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
    nn.Linear(dim, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size, projection_size),
    )
class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector
    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)
    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._encoded
        
def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))

class BYOL(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (96, 96),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.99,
        **hparams,
    ):
        super().__init__()
        self.augment = default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self.hparams = hparams
        self._target = None
        self.encoder(torch.zeros(2, 3, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(x1), self.target(x2)
        loss = (normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)) / 2

        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.target(x1), self.target(x2)
        loss = (normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1)) / 2
        losses1.append(loss)

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())
        
#loading data
dataset=InBreastDataset(csv_file='annotations.csv',root_dir='CLUSTER_INBREAST',transform=
                        transforms.ToTensor())
train_len=int(0.3*len(dataset))
train_unlabel_len=int(0.6*len(dataset))
test_len=len(dataset)-train_len-train_unlabel_len
TRAIN_DATASET,TEST_DATASET,TRAIN_UNLABEL=torch.utils.data.random_split(dataset,[train_len,test_len,train_unlabel_len])


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, **hparams):
        super().__init__(**hparams)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        loss = f.cross_entropy(self.forward(x), y)
        train_loss_list.append(loss)
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        loss = f.cross_entropy(self.forward(x), y)
        losses.append(loss)
        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        val_loss_list.append(val_loss)
        self.log("val_loss", val_loss.item())

model = resnet34(pretrained=True)
model.conv1=torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False) 
batch = torch.rand(4, 1, 224, 224)
model(batch).size()
supervised = SupervisedLightningModule(model)
trainer = pl.Trainer(max_epochs=25, weights_summary=None)
train_dataloader = DataLoader(TRAIN_DATASET, batch_size=128, shuffle=True,drop_last=True,)
test_dataloader = DataLoader(TEST_DATASET, batch_size=128)
trainer.fit(supervised, train_dataloader, test_dataloader)


def accuracy(pred: Tensor, labels: Tensor) -> float:
    return (pred.argmax(dim=-1) == labels).float().mean().item()
acc = sum([accuracy(model1(x), y) for x, y in test_dataloader]) / len(test_dataloader)
print(f"Accuracy: {acc:.3f}")

train_loss_list=[]
val_loss_list=[]
losses=[]
losses1=[]

#saving model
#joblib_file="model_final.pkl"
#joblib.dump(model,joblib_file)

#saving trainer
#trainer_file="trainer_final.pkl"
#joblib.dump(trainer,trainer_file)

model1=joblib.load("C:\\Users\Vedant Manelkar\Desktop\BE_PROJECT\ly_prozect\model_final.pkl")
#joblib.dump(test_dataloader,'dataloader_test.pkl')

test_dataloader=joblib.load("C:\\Users\Vedant Manelkar\Desktop\BE_PROJECT\ly_prozect\dataloader_test.pkl")



#benchmark self supervised models from lightly api
import matplotlib.pyplot as plt


names = ['BarlowTwins', 'DINO', 'Moco','SimCLR','SimSiam','BYOL']
values = [83.5, 86.8,83.8,82.22,77.9,87.2]
plt.figure(figsize=(18, 3))
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('SSL Models')
plt.ylabel('Benchmark accuracy (%)')
plt.show()

#88.7-->92.7,89.7-->92.9,96.7
#Architectures
names1 = ['Resnet-18','Resnet-34','Resnet-50']
values1 = [96.7,92.9,90.7]
plt.figure(figsize=(18, 3))
plt.subplot(131)
plt.bar(names1, values1)
plt.xlabel('Different Resnet architectures')
plt.ylabel('Accuracy (%)')
plt.show()



val_loss_list1=[9.4215,3.9409,2.4928,1.7950,1.7186,0.6382,0.7337,0.4393,0.5672,0.8994,
 0.5141,1.0050,0.5910,0.6063,0.4013,0.3540,0.4507,0.4310,0.4925,0.4302,0.5342,0.7345,0.9614,0.5901,0.4134]
epochs1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
plt.plot(epochs1,val_loss_list1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()