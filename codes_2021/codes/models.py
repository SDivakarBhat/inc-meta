import torch
import numpy as np
import os
import torch.nn as nn
import gensim
from encoder_decoder import UNetWithResnet50Encoder, UNetWithResnet50Decoder
from online_triplet_loss.losses import *
from utils import get_accuracy
from torchnlp.word_to_vector import FastText
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
	def forward(self, x):
		return x.view(x.size(0), -1)


def init_weights(m):
	if type(m)==nn.Linear:
		torch.nn.init.kaiming_normal_(m.weight)
		m.bias.data.fill_(0.01)

def make_fc_1d(f_in, f_out):
	return nn.Sequential(
			nn.Linear(f_in, f_out),
			nn.BatchNorm1d(f_out),
			nn.LeakyReLU(),
			nn.Dropout())

class EmbedBranch(nn.Module):
	def __init__(self, feat_dim, embedding_dim, metric_dim):
		super(EmbedBranch, self).__init__()
		self.fc1 = make_fc_1d(feat_dim, embedding_dim)
		self.fc2 = nn.Linear(embedding_dim, metric_dim)
		self.fc1.apply(init_weights)
		self.fc2.apply(init_weights)
	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = nn.functional.normalize(x)
		return x

class Model(nn.Module):
	def __init__(self, args, num_classes_per_task):
		super(Model, self).__init__()
		self.encoder = UNetWithResnet50Encoder().cuda()
		self.decoder = UNetWithResnet50Decoder().cuda()
		token_dim = args.token_dim
		self.words = nn.Embedding(args.num_classes, token_dim)
		#self.vecs = torch.from_numpy(vecs)
		self.num_classes_per_task = num_classes_per_task
		embedding_dim = args.embedding_dim
		metric_dim = args.metric_dim#embedding_dim//4
		self.args = args
		self.flatten = Flatten()
		self.text_branch = EmbedBranch(token_dim, embedding_dim, metric_dim)
		self.image_branch = EmbedBranch(args.image_feat_dim, embedding_dim, metric_dim)
		self.code2word = EmbedBranch(metric_dim, embedding_dim, metric_dim)
		self.word2vec = Word2Vec(args)
		self.visual2code = EmbedBranch(args.image_feat_dim, embedding_dim, metric_dim)
		self.visual2word_mse = nn.MSELoss()
		self.classifier = Classifier(int(metric_dim*2), args.num_base_classes)	
		self.reconstruction_loss = nn.MSELoss()
		if self.args.dataset =='miniimagenet':
			self.d = {}
			with open("miniimagenet_trainsplit.txt") as f:
    				for line in f:
       					(key, val) = line.split()
       					self.d[str(key)] = str(val)
		#print('d',self.d)	
	def triplet_loss(self, inputs, targets):
		
		return batch_hard_triplet_loss(targets,inputs, margin=0.5, device=self.args.device)

	#def fsl_loss(self):
	#	pass

	#def forward(self):
	#	pass	
	def base_save(self, save_path):
		torch.save({
			
				'decoder':self.decoder.state_dict(),
				'text_branch':self.text_branch.state_dict(),
				'image_branch':self.image_branch.state_dict(),
				'visual2code':self.visual2code.state_dict(),
				'code2word':self.code2word.state_dict(),
				'base_classifier':self.classifier.state_dict()
			
				}, save_path)


	def forward(self, batch):
		##training phase (support)
		train_inputs, train_targets = batch['train']
		train_inputs = train_inputs.cuda()
		train_targets, train_original_labels = train_targets
		train_original_labels = list(zip(*train_original_labels))
		#print(np.shape(train_original_labels))
		#print(train_original_labels[0])#(np.transpose(np.array(train_original_labels))[:,1,:], train_targets)
		if self.args.dataset=='cifar_fs':
			train_vecs = self.word2vec(np.transpose(np.array(train_original_labels)[:,1,:])).cuda()
		elif self.args.dataset=='miniimagenet':
			train_vecs = [self.d[code] for code in np.array(train_original_labels[0])]
			#print(np.shape(np.array(train_vecs)), np.array(train_vecs))
			train_vecs = self.word2vec(np.array(train_vecs)[np.newaxis]).cuda()
		train_triplet_target = torch.cat((train_targets, train_targets))
		train_targets = train_targets.cuda()
		train_triplet_target = torch.reshape(train_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_shots*2,1)).cuda()

		train_visual_bridge, pre_pools  = self.encoder(train_inputs.view(-1, *train_inputs.shape[2:])) #modification required to pass concatenated features through the decoder (separate encoder decoder structure maybe needed)
		#print(train_vecs.shape)
		train_visual_bridge  = torch.reshape(train_visual_bridge,(train_visual_bridge.size(0), train_visual_bridge.size(1)))#train_visual_bridge.permute(0,2,1,3).squeeze(3)#self.flatten(train_visual_bridge).unsqueeze(1)
		#print(train_visual_bridge.shape)
		train_embed_visual = self.image_branch(train_visual_bridge)
		train_embed_text = self.text_branch(train_vecs)	#potential err yle loss calc due2 list type vecs 
		#train_triplet_input = torch.cat((train_embed_visual, train_embed_text))
		#train split triplet loss
		#print(train_triplet_input.is_cuda, train_triplet_target.is_cuda)
		#train_triplet_loss = self.triplet_loss(train_triplet_input.cuda(), train_triplet_target.cuda())

		#print('shape',np.shape(train_embed_visual),np.shape(train_embed_text))
		#concat visual and semantic features
		#testing for v2word#train_concat_features = torch.cat((train_embed_visual, train_embed_text), dim=1) 
		#print('concat_dim', np.shape(train_concat_features))
		visual2code_out_train = self.visual2code(train_visual_bridge)
		visual2word_out_train = self.code2word(visual2code_out_train)
		train_concat_features = torch.cat((train_embed_visual, visual2word_out_train), dim=1) 

		train_triplet_input = torch.cat((train_embed_visual, visual2word_out_train))
		#train_triplet_loss = self.triplet_loss(train_triplet_input.cuda(), train_triplet_target.cuda())
		##testing phase (query)

		test_inputs, test_targets = batch['test']
		test_inputs = test_inputs.to(self.args.device)
		test_targets, test_original_labels = test_targets
		#print(np.shape(test_targets))
		test_original_labels = list(zip(*test_original_labels))
		if self.args.dataset=='cifar_fs':
			test_vecs = self.word2vec(np.transpose(np.array(test_original_labels)[:,1,:])).cuda()
		elif self.args.dataset=='miniimagenet':
			test_vecs = [self.d[code] for code in np.array(test_original_labels[0])]
			test_vecs = self.word2vec(np.array(test_vecs)[np.newaxis]).cuda()
		#test_vecs = self.word2vec(np.transpose(np.array(test_original_labels)[:,1,:])).cuda()
		test_triplet_target = torch.cat((test_targets, test_targets))
		test_targets = test_targets.to(self.args.device)
		test_triplet_target = test_triplet_target.cuda()
		test_triplet_target = torch.reshape(test_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_test_shots*2,1)).cuda()

		test_visual_bridge, test_pre_pools = self.encoder(test_inputs.view(-1, *test_inputs.shape[2:])) #modification required to pass concatenated features through the decoder (separate encoder decoder structure maybe needed)
		test_visual_bridge  = torch.reshape(test_visual_bridge,(test_visual_bridge.size(0), test_visual_bridge.size(1)))
		test_embed_visual = self.image_branch(test_visual_bridge)
		test_embed_text = self.text_branch(test_vecs)	#potential err yle loss calc due2 list type vecs 
		#test_triplet_input = torch.cat((test_embed_visual, test_embed_text))

		#concat visual and semantic featurs
		#testing for v2word#test_concat_features = torch.cat((test_embed_visual, test_embed_text), dim=1) 

		visual2code_out_test = self.visual2code(test_visual_bridge)
		visual2word_out_test = self.code2word(visual2code_out_test)
		test_concat_features = torch.cat((test_embed_visual, visual2word_out_test), dim=1) 

		test_triplet_input = torch.cat((test_embed_visual, visual2word_out_test))
		#test split triplet loss
		#test_triplet_loss = self.triplet_loss(test_triplet_input.cuda(), test_triplet_target.cuda())


		#concatenated featurs thru FC layers
		if self.args.phase =='base':
			train_embeddings = self.classifier(train_concat_features)
			test_embeddings = self.classifier(test_concat_features)
		else:
			train_embeddings = self.test_classifier(train_concat_features)
			test_embeddings = self.test_classifier(test_concat_features)	



		#learn visual to word embedding
		#visual2word_out = self.visual2word(train_visual_bridge)
		visual2word_loss_train = self.visual2word_mse(visual2word_out_train, train_embed_text)

		#visual2word_out = self.visual2word(test_visual_bridge)
		visual2word_loss_test = self.visual2word_mse(visual2word_out_test, test_embed_text)

		"""
		#reconstruct image
		train_decoder_out = self.decoder(train_concat_features.reshape(self.args.batch_size*self.args.num_shots*self.args.num_ways,self.args.image_feat_dim,1,1), pre_pools)
		test_decoder_out = self.decoder(test_concat_features.reshape(self.args.batch_size*self.args.num_test_shots*self.args.num_ways,self.args.image_feat_dim,1,1), test_pre_pools)
		#reconstruction loss
		train_recon_loss = self.reconstruction_loss(train_decoder_out.unsqueeze(0),train_inputs)
		test_recon_loss = self.reconstruction_loss(test_decoder_out.unsqueeze(0), test_inputs)
		"""
		#fsl
		#print('prot',np.shape(test_embeddings.unsqueeze(0)),np.shape(test_targets), np.shape(self.num_classes_per_task))
		prototypes = get_prototypes(train_embeddings.unsqueeze(0), train_targets, self.num_classes_per_task)
		#print('proto',np.shape(prototypes))
		fsl_loss = prototypical_loss(prototypes, test_embeddings.unsqueeze(0), test_targets)
		

		##total loss
		loss = fsl_loss + visual2word_loss_train + visual2word_loss_test  #train_triplet_loss + test_triplet_loss + visual2word_loss_train + visual2word_loss_test #+ train_recon_loss + test_recon_loss
		

		##accuracy calculation
		accuracy, log_py = get_accuracy(prototypes, test_embeddings.unsqueeze(0), test_targets)
		#print(np.array(test_original_labels)[:,1,:])
		del(batch)
		del(train_inputs)
		del(test_inputs)
		del(train_targets)
		del(test_targets)	
		return loss, accuracy, log_py, visual2word_loss_train, visual2word_loss_test#, train_recon_loss, test_recon_loss
 

class Word2Vec(nn.Module):

	def __init__(self,args):
		super(Word2Vec, self).__init__()
		self.args = args
		self.word2vec = fasttext.load_facebook_vectors('/home/SharedData/Divakar/project2/data/FastText/gensim/wiki.en.bin')#FastText(cache='/home/SharedData/Divakar/project2/data/FastText') #gensim.models.KeyedVectors.load_word2vec_format('/home/SharedData/Divakar/cvpr/GoogleNews-vectors-negative300.bin', binary=True)
		
	def forward(self, words):
		#words = words.reshape(self.args.batch_size,1)
		#print('shape',words, words[0],np.shape(words), np.shape(words[0]))
		words = np.array(words)
		m,n = np.shape(words)
		#words = np.array(words)
		#m = self.args.batch_size
		#n = self.args.num_ways*self.args.num_shots
		vecs = torch.empty(m*n, self.args.token_dim)
		words = words.reshape(1,m*n)
		#print('here',np.shape(words), np.shape(words[0]), words, words[0])
		for i,x in enumerate(words[0]):
			v = self.word2vec[x].copy()
			vecs[i] = torch.from_numpy(v)
		return vecs
	

class embedding(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super(embedding,self).__init__()
		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.encoder = nn.Sequential(
				nn.Linear(self.in_dim, self.hid_dim),
				nn.LeakyReLU(),
				nn.Dropout(),
				nn.Linear(self.hid_dim, self.hid_dim),
				nn.LeakyReLU(),
				nn.Dropout(),
				nn.Linear(self.hid_dim, self.hid_dim),
				nn.LeakyReLU(),
				nn.Dropout(),
				nn.Linear(self.hid_dim, self.out_dim),
				nn.LeakyReLU()
		)

		self.encoder.apply(init_weights)
	def forward(self, x):
		return self.encoder(x)

class Classifier(nn.Module):
	def __init__(self, in_dim,  n_classes):
		super(Classifier, self).__init__()
		self.n_classes = n_classes
		out_dim = n_classes
		self.classifier = nn.Sequential(
					nn.Linear(in_dim, in_dim//2),
					nn.LeakyReLU(),
					nn.Dropout(),
					nn.Linear(in_dim//2, in_dim//4),
					nn.LeakyReLU(),
					nn.Dropout(),
					nn.Linear(in_dim//4, out_dim),
					nn.LeakyReLU()	
		)	
		self.classifier.apply(init_weights)
	def forward(self, x):
		return self.classifier(x)

			

