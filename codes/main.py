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



class CategoricalAndLabels(Categorical):
	def __call__(self, target):
		label, class_augmentaion = target
		return (self.classes[target],label)

def train(dataloader, Model, log_dir):
	#params = list(Model.Classifier.parameters())+list(Model.decoder.parameters())	
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr = args.lr, weight_decay= args.wd)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
	epoch = 0
	writer = SummaryWriter(log_dir=log_dir)	
	while(epoch < args.max_epoch):
		model.train()
		#scheduler.step()
		meter = []
		with tqdm(dataloader, total=args.max_episode, desc='Epoch {:d}'.format(epoch+1)) as pbar:
			for idx,sample in enumerate(pbar):
				optimizer.zero_grad()
				loss, accuracy = Model(sample)
				loss.backward()
				optimizer.step()
				meter.append(accuracy)
				if idx>=args.max_episode:
					break
		scheduler.step()
		print("Epoch {:0.2f} accuracy is {:0.2f}".format(epoch+1,(sum(meter)/len(meter))))
		epoch += 1
		writer.add_scalar('Accuracy',(sum(meter)/len(meter)),epoch)

if __name__=='__main__':
	
	import argparse
	
	parse = argparse.ArgumentParser('Project2')

	parse.add_argument('--dataset', type=str, default='cifar_fs')
	parse.add_argument('--phase', type=str, default='base')	
	parse.add_argument('--data_folder',type=str, default='/home/SharedData/Divakar/project2/data',help='path to the data roo folder')
	parse.add_argument('--log_dir',type=str, default='/home/SharedData/Divakar/project2/log',help='path to the log roo folder')

	parse.add_argument('--num_shots', type=int, default=5, help='number of samples per class per episode in train split')
	parse.add_argument('--num_test_shots', type=int, default=15, help='number of samples per class per episode in test split')

	parse.add_argument('--num_ways', type=int, default=5, help='number of classes per episode')
	parse.add_argument('--download', type=bool, default=True, help='whether the dataset is to be downloaded if absent in the root data folder')
	parse.add_argument('--batch_size', type=int, default=1, help='batch size')
	parse.add_argument('--num_workers', type=int, default=1)
	parse.add_argument('-use_cuda', type=bool, default=True)
	parse.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parse.add_argument('--wd', type=float, default=5e-4, help='weight decay')
	parse.add_argument('--max_epoch', type=int, default=1000)
	parse.add_argument('--max_episode',type=int, default=100)
	parse.add_argument('--step_size',type=int, default=20)
	parse.add_argument('--gamma', type=float, default=0.5)
	parse.add_argument('--token_dim',type=int, default=300, help='dimension of word features')
	parse.add_argument('--num_base_classes', type=int, default=64)
	parse.add_argument('--embedding_dim', type=int, default=512)
	parse.add_argument('--metric_dim', type=int, default=1024)
	parse.add_argument('--num_classes', type=int, default=100)
	parse.add_argument('--image_feat_dim', type=int, default=2048)
	parse.add_argument('--device',type=str, default='cuda')
	parse.add_argument('--log_id',type=str)
	args = parse.parse_args()
	#word2vec = Word2Vec() 
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
		log_dir = args.log_dir+'/{}'.format(args.dataset)
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
		log_dir = args.log_dir + '/{}'.format(args.dataset)
	train_dataloader = BatchMetaDataLoader(
					dataset,
					batch_size=args.batch_size,
					shuffle=True,
					num_workers=args.num_workers)
	model = Model(args, dataset.num_classes_per_task).cuda() 

	train(train_dataloader, model, log_dir)

				
