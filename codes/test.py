import torch
import torch.nn as nn
from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from models_modified import Model, Word2Vec
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

def test(train_dataloader, test_dataloader, Model, log_dir, save_dir):
	#params = list(Model.Classifier.parameters())+list(Model.decoder.parameters())	
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr = args.lr, weight_decay= args.wd)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
	epoch = 0
	writer = SummaryWriter(log_dir=log_dir)	
	while(epoch < args.max_epoch):
		model.train()
		#scheduler.step()
		meter = []
		entrpy = []
		base_entrpy = []
		with tqdm(test_dataloader, total=args.max_episode, desc='Epoch {:d}'.format(epoch+1)) as pbar:
			for idx,sample in enumerate(pbar):
				optimizer.zero_grad()
				loss, accuracy, logpy, dummy_logpy = Model(sample)
				loss.backward()
				optimizer.step()
				meter.append(accuracy)
				entrpy.append(logpy)
				base_entrpy.append(dummy_logpy)
				if idx>=args.max_episode:
					break
		scheduler.step()
		print("Epoch {:0.2f} target accuracy is {:0.2f}".format(epoch+1,(sum(meter)/len(meter))))
		epoch += 1
		writer.add_scalar('Test Accuracy',(sum(meter)/len(meter)),epoch)
		writer.add_scalar('Base CEntropy',(sum(base_entrpy)/len(base_entrpy)),epoch)
		writer.add_scalar('Target CEntropy',(sum(entrpy)/len(entrpy)),epoch)
	Model.target_save(save_dir)
if __name__=='__main__':
	
	import argparse
	
	parse = argparse.ArgumentParser('Project2')

	parse.add_argument('--dataset', type=str, default='cifar_fs')
	parse.add_argument('--phase', type=str, default='test')	
	parse.add_argument('--data_folder',type=str, default='/home/SharedData/Divakar/project2/data',help='path to the data roo folder')
	parse.add_argument('--log_dir',type=str, default='/home/SharedData/Divakar/project2/log',help='path to the log roo folder')
	parse.add_argument('--save_dir',type=str, default='/home/SharedData/Divakar/project2/saved',help='path to the log roo folder')

	parse.add_argument('--num_shots', type=int, default=5, help='number of samples per class per episode in train split')
	parse.add_argument('--num_test_shots', type=int, default=15, help='number of samples per class per episode in test split')

	parse.add_argument('--num_ways', type=int, default=5, help='number of classes per episode')
	parse.add_argument('--download', type=bool, default=True, help='whether the dataset is to be downloaded if absent in the root data folder')
	parse.add_argument('--batch_size', type=int, default=1, help='batch size')
	parse.add_argument('--num_workers', type=int, default=1)
	parse.add_argument('-use_cuda', type=bool, default=True)
	parse.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parse.add_argument('--wd', type=float, default=5e-4, help='weight decay')
	parse.add_argument('--max_epoch', type=int, default=500)
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
	parse.add_argument('--log_id',type=str)
	args = parse.parse_args()
	#word2vec = Word2Vec() 
	if args.dataset=='cifar_fs':
		train_dataset = cifar_fs(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_train=True,
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)

		test_dataset = cifar_fs(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_test=True,
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)
		log_dir = args.log_dir+'/{}'.format(args.dataset)+args.phase+args.log_id
		old_save_dir =  args.save_dir+'/{}'.format(args.dataset)+'base'+args.log_id+'.pth'
		save_dir = args.save_dir+'/{}'.format(args.dataset)+args.phase+args.log_id+'.pth'

	elif args.dataset=='miniimagenet':
		train_dataset = miniimagenet(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_train=True,
				transform=Compose([Resize(32), ToTensor()]),
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)


		test_dataset = miniimagenet(
				args.data_folder,
				shots=args.num_shots,
				ways=args.num_ways,
				shuffle=True,
				test_shots=15,
				meta_test=True,
				transform=Compose([Resize(32), ToTensor()]),
				target_transform=CategoricalAndLabels(num_classes=5),
				download=args.download)
		log_dir = args.log_dir + '/{}'.format(args.dataset)+args.phase+args.log_id
		old_save_dir =  args.save_dir+'/{}'.format(args.dataset)+'base'+args.log_id+'.pth'
		save_dir = args.save_dir+'/{}'.format(args.dataset)+args.phase+args.log_id+'.pth'

	train_dataloader = BatchMetaDataLoader(
					train_dataset,
					batch_size=args.batch_size,
					shuffle=True,
					num_workers=args.num_workers)
	test_dataloader = BatchMetaDataLoader(
					test_dataset,
					batch_size=args.batch_size,
					shuffle=True,
					num_workers=args.num_workers)
	model = Model(args, test_dataset.num_classes_per_task, old_save_dir).cuda() 

	test(train_dataloader, test_dataloader, model, log_dir, save_dir)


