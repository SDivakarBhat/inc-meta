import torch
import numpy as np
import torch.nn.functional as F

def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(2)
        - embeddings.unsqueeze(1)) ** 2, dim=-1)
    #logpy =  F.cross_entropy(-squared_distances, targets, **kwargs)
    #print('dist',np.shape(sq_distances))
    _, predictions = torch.min(sq_distances, dim=-2)
    #print('pred shape',np.shape(predictions),predictions)
    acc = torch.eq(predictions,targets.squeeze()).float().mean()
    #print('acc',acc)
    return torch.mean(predictions.eq(targets).float())


