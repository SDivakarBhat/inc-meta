import torch
import torch.nn as nn
from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from models import Model
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
# def train_embedding():
# import random


class CategoricalAndLabels(Categorical):
    """
    Returns categorical class and label
    """

    def __call__(self, target):
        label, class_augmentaion = target
        return (self.classes[target], label)


def train(dataloader, model, log_dir, save_dir, args_path):
    """
    Training function
    """

    # params = list(Model.Classifier.parameters())+
    #               list(Model.decoder.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        Model.parameters()), lr=ARGS.lr,
                                 weight_decay=ARGS.wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ARGS.step_size,
    #                                            gamma=ARGS.gamma)
    epoch = 0
    writer = SummaryWriter(log_dir=log_dir)
    F = open(args_path, "a")
    while epoch < ARGS.max_epoch:
        model.train()

        meter = []
        entrpy = []
        # trip1_ = []
        # trip2_ = []
        v2w1_ = []
        v2w2_ = []
        # recon1_ =[]
        # recon2_ =[]
        with tqdm(dataloader, total=ARGS.max_episode,
                  desc='Epoch {:d}'.format(epoch+1)) as pbar:
            for idx, sample in enumerate(pbar):
                optimizer.zero_grad()
                loss, accuracy, logpy, v2w1, v2w2 = model(sample)
                loss.backward()
                optimizer.step()
                meter.append(accuracy)
                entrpy.append(logpy)
                # trip1_.append(trip1)
                # trip2_.append(trip2)
                v2w1_.append(v2w1)
                v2w2_.append(v2w2)
                # recon1_.append(recon1)
                # recon2_.append(recon2)
                if idx >= ARGS.max_episode:
                    break
        # scheduler.step()
        # random.shuffle(dataloader)
        print("Epoch {:0.2f} accuracy is {:0.2f}"
              .format(epoch+1, (sum(meter)/len(meter))))
        epoch += 1
        writer.add_scalar('Accuracy', (sum(meter)/len(meter)), epoch)
        # writer.add_scalar('CEntropy', (sum(entrpy)/len(entrpy)),epoch)
        # writer.add_scalar('triplet train',(sum(trip1_)/len(trip1_)),epoch)
        # writer.add_scalar('triplet test',(sum(trip2_)/len(trip2_)),epoch)
        writer.add_scalar('v2w train', (sum(v2w1_)/len(v2w1_)), epoch)
        writer.add_scalar('v2w_test', (sum(v2w2_)/len(v2w2_)), epoch)
        # writer.add_scalar('recon train',(sum(recon1_)/len(recon1_)),epoch)
        f.write("Epoch {:0.2f} accuracy is {:0.2f}\n"
                .format(epoch+1, (sum(meter)/len(meter))))
        model.base_save(save_dir)
    F.close()


if __name__ == '__main__':
    import argparse
    PARSE = argparse.ArgumentParser('Project2')
    PARSE.add_argument('--dataset', type=str, default='cifar_fs')
    PARSE.add_argument('--phase', type=str, default='base')
    PARSE.add_argument('--data_folder',
                       type=str,
                       default='/home/SharedData/Divakar/project2/data',
                       help='path to the data roo folder')
    PARSE.add_argument('--log_dir',
                       type=str,
                       default='/home/SharedData/Divakar/project2/log',
                       help='path to the log roo folder')
    PARSE.add_argument('--save_dir',
                       type=str,
                       default='/home/SharedData/Divakar/project2/saved',
                       help='path to the log root folder')

    PARSE.add_argument('--num_shots',
                       type=int, default=5,
                       help='number of samples/class/episode in train split')
    PARSE.add_argument('--num_test_shots',
                       type=int,
                       default=15,
                       help='number of samples/class/episode in test split')

    PARSE.add_argument('--num_ways',
                       type=int,
                       default=5, help='number of classes per episode')
    PARSE.add_argument('--download',
                       type=bool, default=True,
                       help='if dataset is tobe downloaded when absent')
    PARSE.add_argument('--batch_size', type=int, default=1, help='batch size')
    PARSE.add_argument('--num_workers', type=int, default=1)
    PARSE.add_argument('-use_cuda', type=bool, default=True)
    PARSE.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    PARSE.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    PARSE.add_argument('--max_epoch', type=int, default=1000)
    PARSE.add_argument('--max_episode', type=int, default=100)
    PARSE.add_argument('--step_size', type=int, default=20)
    PARSE.add_argument('--gamma', type=float, default=0.5)
    PARSE.add_argument('--token_dim', type=int, default=300,
                       help='dimension of word features')
    PARSE.add_argument('--num_base_classes', type=int, default=64)
    PARSE.add_argument('--num_target_classes', type=int, default=20)
    PARSE.add_argument('--embedding_dim', type=int, default=512)
    PARSE.add_argument('--metric_dim', type=int, default=1024)
    PARSE.add_argument('--num_classes', type=int, default=100)
    PARSE.add_argument('--image_feat_dim', type=int, default=2048)
    PARSE.add_argument('--device', type=str, default='cuda')
    PARSE.add_argument('--log_id', type=str)
    PARSE.add_argument('--resume', type=str, default='False')
    ARGS = PARSE.parse_args()
    # word2vec = Word2Vec()
    print(ARGS)
    args_path = ARGS.log_dir+'/{}_{}.txt'.format(ARGS.dataset, ARGS.log_id)
    f = open(args_path, "w")
    f.write(str(ARGS))
    f.close()
    if ARGS.dataset == 'cifar_fs':
        DATASET = cifar_fs(
                        ARGS.data_folder,
                        shots=ARGS.num_shots,
                        ways=ARGS.num_ways,
                        shuffle=True,
                        test_shots=15,
                        meta_train=True,
                        target_transform=CategoricalAndLabels(num_classes=5),
                        download=ARGS.download)
        log_dir = ARGS.log_dir+'/{}'.format(ARGS.dataset)+ARGS.phase+ARGS.log_id
        save_dir = ARGS.save_dir+'/{}'.format(ARGS.dataset)+ARGS.phase+ARGS.log_id+'.pth'
    elif ARGS.dataset == 'miniimagenet':
        DATASET = miniimagenet(
                        ARGS.data_folder,
                        shots=ARGS.num_shots,
                        ways=ARGS.num_ways,
                        shuffle=True,
                        test_shots=15,
                        meta_train=True,
                        transform=Compose([Resize(32), ToTensor()]),
                        target_transform=CategoricalAndLabels(num_classes=5),
                        download=ARGS.download)
        log_dir = ARGS.log_dir + '/{}'.format(ARGS.dataset)+ARGS.phase+ARGS.log_id
        save_dir = ARGS.save_dir+'/{}'.format(ARGS.dataset)+ARGS.phase+ARGS.log_id+'.pth'

    TRAIN_DATALOADER = BatchMetaDataLoader(DATASET,
                                           batch_size=ARGS.batch_size,
                                           shuffle=True,
                                           num_workers=ARGS.num_workers)
    MODEL = Model(ARGS. DATASET.num_classes_per_task).cuda()

    train(TRAIN_DATALOADER, MODEL, log_dir, save_dir, args_path)
