from torchmeta.datasets import Omniglot, CIFARFS
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
import torch
from tqdm import tqdm
import numpy as np
from models import Word2Vec

class CategoricalAndLabels(Categorical):
      def __call__(self, target):
          label, class_augmentaion = target
          return (self.classes[target],label)

dataset = CIFARFS("/home/SharedData/Divakar/project2/data",
                   # Number of ways
                   num_classes_per_task=5,
                   # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                   target_transform=CategoricalAndLabels(num_classes=5),
                   # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   #target_transform = CategoricalAndLabels(num_classes=5),
                   download=True)
dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
class_labels = []
word2vec = Word2Vec()
print(dataset.num_classes_per_task)
with tqdm(dataloader,total=100) as pbar:
 for batch_idx, batch in enumerate(pbar):

    train_inputs, train_targets = batch["train"]
    #print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    #print('Train targets : {0}'.format(torch.unique(train_targets)))  # (16, 25)
    #print(batch)
    #class_labels.append(torch.unique(train_targets))
    train_targets, original_labels = train_targets
    original_labels = list(zip(*original_labels))
    vecs = word2vec(original_labels[0][1])#[word2vec[word] for word in original_labels[0][1]]
    print(f'original labels:\n{original_labels[0][1]}')
    print('vecs {}'.format(vecs))
    test_inputs, test_targets = batch["test"]
    #print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    #print('Test targets : {0}'.format(torch.unique(test_targets)))
    #class_labels.append(torch.unique(test_targets))
    if batch_idx >= 100:
        break

#print(torch.unique(class_labels))
