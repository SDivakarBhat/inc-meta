import torch
import numpy as np
from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from models_augmented_test import Model
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
# def train_embedding():
# import random
from augmenter_cgan import Generator, Discriminator

class CategoricalAndLabels(Categorical):
    """
    Returns categorical class and label
    """

    def __call__(self, target):
        label, class_augmentaion = target
        return (self.classes[target], label)


def test(train_dataloader, test_dataloader, model, gen, log_dir):
    """
    Training function
    """

    # params = list(Model.Classifier.parameters())\
    #          +list(Model.decoder.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                 model.parameters()),
                                 lr=ARGS.lr,
                                 weight_decay=ARGS.wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ARGS.step_size,
    #                                             gamma=ARGS.gamma)
    epoch = 0
    writer = SummaryWriter(log_dir=log_dir)
    # dataloader = list(zip(train_dataloader, test_dataloader))
    # random.shuffle(dataloader)
    while epoch < ARGS.max_epoch:
        model.train()
        # scheduler.step()
        meter = []
        # entrpy = []
        meter_base = []
        # with torch.no_grad():
        with tqdm(test_dataloader, total=ARGS.max_episode) as pbar:
            for idx, sample in enumerate(pbar):
                optimizer.zero_grad()
                accuracy, loss = model(sample)
                # loss.backward()
                # optimizer.step()

                imgs, lbls = sample['train']
                batch_size = np.shape(imgs.view(-1, *imgs.shape[2:]))[0]
                z_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, ARGS.zdim))) 
                labels = np.random.choice(ARGS.num_ways, ARGS.num_ways, replace=False)               
                gen_labels = torch.cuda.LongTensor([x for x in labels for i in range(ARGS.num_shots)]).cuda()
                # print('genlbl', gen_labels)
                fake = gen(z_noise, gen_labels)
                generated_train = (fake, gen_labels)

                imgs, lbls = sample['test']
                batch_size = np.shape(imgs.view(-1, *imgs.shape[2:]))[0]
                # print(batch_size)
                z_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, ARGS.zdim)))                
                # gen_labels_test = torch.cuda.LongTensor(np.random.randint(0, ARGS.num_base_classes, batch_size)).cuda()
                # labels = torch.unique(gen_labels)
                # print(labels)
                gen_labels_test = torch.cuda.LongTensor([x for x in labels for i in range(ARGS.num_test_shots)])
                fake_test = gen(z_noise, gen_labels_test)
                generated_test = (fake_test, gen_labels_test)

                generated = (fake, gen_labels, fake_test, gen_labels_test)
                # generated = generated.cuda()
                base_accuracy, loss_fake = model(generated, flag='fake')
                loss_total = loss + loss_fake
                loss_total.backward()
                optimizer.step()

                meter_base.append(base_accuracy)

                meter.append(accuracy)
                # entrpy.append(lp)
                # base_entrpy.append(blp)
                if idx >= ARGS.max_episode:
                    break
        target_accuracy = sum(meter)/len(meter)
        base_accuracy = sum(meter_base)/len(meter_base)
        accuracy = (target_accuracy+base_accuracy)/2
        print("Epoch {} of evaluation, accuracy={}".format(epoch+1,
                                                            accuracy))
        """
        with tqdm(train_dataloader, total=ARGS.max_episode) as pbar:
            for idx, sample in enumerate(pbar):
                optimizer.zero_grad()
                imgs, lbls = sample['train']
                batch_size = np.shape(imgs.view(-1, *imgs.shape[2:]))[0]
                z_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, args.zdim)))                
                gen_labels = torch.cuda.LongTensor(np.random.randint(0, args.num_base_classes, batch_size)).cuda()
                fake = gen(z_noise, gen_labels)
                accuracy, loss = model(sample)
                loss.backward()
                optimizer.step()
                generated = (fake, gen_labels)
                accuracy, loss = model(generated, flag='fake')
                loss.backward()
                optimizer.step()

                meter_base.append(accuracy)
                # entrpy_base.append(lp)
                # base_entrpy.append(blp)
                if idx >= ARGS.max_episode:
                    break
        base_accuracy = sum(meter_base)/len(meter_base)
        # scheduler.step()
        print("Epoch {} of train loader, accuracy={}".format(epoch+1,
                                                             base_accuracy))
        """
        epoch += 1
    print("Eval accuracy is {:0.2f}".format(accuracy))
    # epoch += 1
    writer.add_scalar('Eval Accuracy', ((base_accuracy+target_accuracy)/2))
    # writer.add_scalar('Base CEntropy',
    #                   (sum(base_entrpy)/len(base_entrpy)),epoch)
    # writer.add_scalar('Target CEntropy',(sum(entrpy)/len(entrpy)),epoch)
    # model.target_save(save_dir)


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser('Project2')
    parse.add_argument('--dataset', type=str, default='cifar_fs')
    parse.add_argument('--phase', type=str, default='eval')
    parse.add_argument('--data_folder',
                       type=str,
                       default='/home/SharedData/Divakar/project2/data',
                       help='path to the data roo folder')
    parse.add_argument('--log_dir',
                       type=str,
                       default='/home/SharedData/Divakar/project2/log',
                       help='path to the log roo folder')
    parse.add_argument('--save_dir',
                       type=str,
                       default='/home/SharedData/Divakar/project2/saved',
                       help='path to the log roo folder')

    parse.add_argument('--num_shots',
                       type=int,
                       default=5,
                       help='number of samples per class \
                               per episode in train split')
    parse.add_argument('--num_test_shots',
                       type=int,
                       default=15,
                       help='number of samples per class \
                               per episode in test split')

    parse.add_argument('--num_ways',
                       type=int,
                       default=5,
                       help='number of classes per episode')
    parse.add_argument('--download',
                       type=bool,
                       default=True,
                       help='whether the dataset is to be \
                       downloaded if absent in the root data folder')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--num_workers', type=int, default=1)
    parse.add_argument('-use_cuda', type=bool, default=True)
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parse.add_argument('--max_epoch', type=int, default=50)
    parse.add_argument('--max_episode', type=int, default=100)
    parse.add_argument('--step_size', type=int, default=20)
    parse.add_argument('--gamma', type=float, default=0.5)
    parse.add_argument('--token_dim', type=int, default=300,
                       help='dimension of word features')
    parse.add_argument('--num_base_classes', type=int, default=64)
    parse.add_argument('--num_target_classes', type=int, default=20)
    parse.add_argument('--embedding_dim', type=int, default=512)
    parse.add_argument('--metric_dim', type=int, default=1024)
    parse.add_argument('--num_classes', type=int, default=100)
    parse.add_argument('--image_feat_dim', type=int, default=2048)
    parse.add_argument('--zdim', type=int, default=100)
    parse.add_argument('--device', type=str, default='cuda')
    parse.add_argument('--log_id', type=str)
    ARGS = parse.parse_args()
    print(ARGS)
    # args_path=ARGS.log_dir='/{}_{}.txt'.format(ARGS.dataset,ARGS.log_id)
    # f = open(args_path,'a')
    # f.write('Evaluation (Target data tuning)')
    # word2vec = Word2Vec()
    if ARGS.dataset == 'cifar_fs':
        TRAIN_DATASET = cifar_fs(ARGS.data_folder,
                                 shots=ARGS.num_shots,
                                 ways=ARGS.num_ways,
                                 shuffle=True,
                                 test_shots=15,
                                 meta_train=True,
                                 target_transform=CategoricalAndLabels(num_classes=5),
                                 download=ARGS.download)

        TEST_DATASET = cifar_fs(
                        ARGS.data_folder,
                        shots=ARGS.num_shots,
                        ways=ARGS.num_ways,
                        shuffle=True,
                        test_shots=15,
                        meta_test=True,
                        target_transform=CategoricalAndLabels(num_classes=5),
                        download=ARGS.download)
        LOG_DIR = ARGS.log_dir + '/{}'.format(ARGS.dataset) + \
                                 ARGS.phase + ARGS.log_id
        OLD_SAVE_DIR = ARGS.save_dir + '/{}'.format(ARGS.dataset)\
                                     + 'base' + ARGS.log_id + '.pth'
        # save_dir = ARGS.save_dir+'/{}'.format(ARGS.dataset)+\
        #                        ARGS.phase+ARGS.log_id+'.pth'

    elif ARGS.dataset == 'miniimagenet':
        TRAIN_DATASET = miniimagenet(ARGS.data_folder,
                                     shots=ARGS.num_shots,
                                     ways=ARGS.num_ways,
                                     shuffle=True,
                                     test_shots=15,
                                     meta_train=True,
                                     transform=Compose([Resize(32),
                                                        ToTensor()]),
                                     target_transform=CategoricalAndLabels(num_classes=5),
                                     download=ARGS.download)

        TEST_DATASET = miniimagenet(ARGS.data_folder,
                                    shots=ARGS.num_shots,
                                    ways=ARGS.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_test=True,
                                    transform=Compose([Resize(32),
                                                       ToTensor()]),
                                    target_transform=CategoricalAndLabels(num_classes=5),
                                    download=ARGS.download)
        LOG_DIR = ARGS.log_dir + '/{}'.format(ARGS.dataset)\
                               + ARGS.phase + ARGS.log_id
        OLD_SAVE_DIR = ARGS.save_dir + '/{}'.format(ARGS.dataset)\
                                     + 'base' + ARGS.log_id+'.pth'
        # save_dir = ARGS.save_dir+'/{}'.format(ARGS.dataset)\
        #                         +ARGS.phase+ARGS.log_id+'.pth'
    GEN_PATH = ARGS.save_dir + '/{}'.format(ARGS.dataset)\
                                     + 'base' + ARGS.log_id+'_GEN.pth'

    TRAIN_DATALOADER = BatchMetaDataLoader(TRAIN_DATASET,
                                           batch_size=ARGS.batch_size,
                                           shuffle=True,
                                           num_workers=ARGS.num_workers)
    TEST_DATALOADER = BatchMetaDataLoader(TEST_DATASET,
                                          batch_size=ARGS.batch_size,
                                          shuffle=True,
                                          num_workers=ARGS.num_workers)
    # print('ii', TEST_DATASET.num_classes_per_task)
    MODEL = Model(ARGS, TEST_DATASET.num_classes_per_task, OLD_SAVE_DIR).cuda()
    GEN = Generator(ARGS).cuda()
    GEN.load_state_dict(torch.load(GEN_PATH))
    GEN.eval()
    test(TRAIN_DATALOADER, TEST_DATALOADER, MODEL, GEN, LOG_DIR)
