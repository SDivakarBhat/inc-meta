import torch
import torch.nn as nn
from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from models import Model, Word2Vec
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
#def train_embedding():
import random
import os
import numpy as np

class CategoricalAndLabels(Categorical):
	def __call__(self, target):
		label, class_augmentaion = target
		return (self.classes[target],label)

def train(dataloader, Model, log_dir, save_dir, args_path):
	#params = list(Model.Classifier.parameters())+list(Model.decoder.parameters())	
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr = args.lr, weight_decay= args.wd)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
	epoch = 0
	writer = SummaryWriter(log_dir=log_dir)	
	f = open(args_path,"a")
	while(epoch < args.max_epoch):
		Model.train()

		meter = []
		entrpy = []
		trip1_ = []
		trip2_ = []
		v2w1_ = []
		v2w2_ = []
		recon1_ =[]
		recon2_ =[]
		with tqdm(dataloader, total=args.max_episode, desc='Epoch {:d}'.format(epoch+1)) as pbar:
			for idx, sample in enumerate(pbar):
				optimizer.zero_grad()
				loss, accuracy, logpy,v2w1,v2w2 = Model(sample)
				loss.backward()
				optimizer.step()
				meter.append(accuracy)
				entrpy.append(logpy)
				#trip1_.append(trip1)
				#trip2_.append(trip2)
				v2w1_.append(v2w1)
				v2w2_.append(v2w2)
				#recon1_.append(recon1)
				#recon2_.append(recon2)
				if idx>=args.max_episode:
					break
		#scheduler.step()
		#random.shuffle(dataloader)
		print("Epoch {:0.2f} accuracy is {:0.2f}".format(epoch+1,(sum(meter)/len(meter))))
		epoch += 1
		writer.add_scalar('Accuracy',(sum(meter)/len(meter)),epoch)
		#writer.add_scalar('CEntropy',(sum(entrpy)/len(entrpy)),epoch)
		#writer.add_scalar('triplet train',(sum(trip1_)/len(trip1_)),epoch)
		#writer.add_scalar('triplet test',(sum(trip2_)/len(trip2_)),epoch)
		writer.add_scalar('v2w train',(sum(v2w1_)/len(v2w1_)),epoch)
		writer.add_scalar('v2w_test',(sum(v2w2_)/len(v2w2_)),epoch)
		#writer.add_scalar('recon train',(sum(recon1_)/len(recon1_)),epoch)
		f.write("Epoch {:0.2f} accuracy is {:0.2f}\n".format(epoch+1,(sum(meter)/len(meter))))
		
	Model.base_save(save_dir)
	f.close()
if __name__=='__main__':
	
	import argparse
	
	parse = argparse.ArgumentParser('Project2')

	parse.add_argument('--dataset', type=str, default='cifar_fs')
	parse.add_argument('--phase', type=str, default='base')	
	parse.add_argument('--data_folder',type=str, default='/home/SharedData/Divakar/project2/data',help='path to the data roo folder')
	parse.add_argument('--log_dir',type=str, default='/home/SharedData/Divakar/project2/log',help='path to the log roo folder')
	parse.add_argument('--save_dir',type=str, default='/home/SharedData/Divakar/project2/saved',help='path to the log root folder')

	parse.add_argument('--num_shots', type=int, default=5, help='number of samples per class per episode in train split')
	parse.add_argument('--num_test_shots', type=int, default=15, help='number of samples per class per episode in test split')

	parse.add_argument('--num_ways', type=int, default=5, help='number of classes per episode')
	parse.add_argument('--download', type=bool, default=True, help='whether the dataset is to be downloaded if absent in the root data folder')
	parse.add_argument('--batch_size', type=int, default=1, help='batch size')
	parse.add_argument('--num_workers', type=int, default=1)
	parse.add_argument('-use_cuda', type=bool, default=True)
	parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parse.add_argument('--wd', type=float, default=1e-4, help='weight decay')
	parse.add_argument('--max_epoch', type=int, default=1000)
	parse.add_argument('--max_episode',type=int, default=100)
	parse.add_argument('--step_size',type=int, default=20)
	parse.add_argument('--gamma', type=float, default=0.5)
	parse.add_argument('--token_dim',type=int, default=300, help='dimension of word features')
	parse.add_argument('--num_base_classes', type=int, default=64)
	parse.add_argument('--num_target_classes', type=int, default=20)
	parse.add_argument('--embedding_dim', type=int, default=512)
	parse.add_argument('--metric_dim', type=int, default=1024)
	parse.add_argument('--num_classes', type=int, default=100)
	parse.add_argument('--image_feat_dim', type=int, default=2048)
	parse.add_argument('--device',type=str, default='cuda')
	parse.add_argument('--gpu', type=str, default='1')
	parse.add_argument('--seed', type=int, default='0')
	parse.add_argument('--log_id',type=str)
	parse.add_argument('--resume', type=str, default='False')
	args = parse.parse_args()
	#word2vec = Word2Vec()
	print(args)
	args_path = args.log_dir+'/{}_{}.txt'.format(args.dataset,args.log_id)
	f = open(args_path,"w")
	f.write(str(args))
	f.close()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device(args.device)#torch.device('cuda' if use_cuda else 'cpu')pen(ARGS_PATH, "w")
	if args.dataset=='cifar_fs':
		dataset = cifar_fs(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_train=True,
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)
		log_dir = args.log_dir+'/{}'.format(args.dataset)+args.phase+args.log_id
		save_dir = args.save_dir+'/{}'.format(args.dataset)+args.phase+args.log_id+'.pth'
	elif args.dataset=='miniimagenet':
		dataset = miniimagenet(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_train=True,
				transform=Compose([Resize(32), ToTensor()]),
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)
		log_dir = args.log_dir + '/{}'.format(args.dataset)+args.phase+args.log_id
		save_dir = args.save_dir+'/{}'.format(args.dataset)+args.phase+args.log_id+'.pth'

	train_dataloader = BatchMetaDataLoader(
					dataset,
					batch_size=args.batch_size,
					shuffle=True,
					num_workers=args.num_workers)
	model = Model(args, dataset.num_classes_per_task).cuda() 

	train(train_dataloader, model, log_dir, save_dir, args_path)

				
