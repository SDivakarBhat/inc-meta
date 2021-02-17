import torch
import numpy as np
# import os
import torch.nn as nn
# import gensim
from encoder_decoder import UNetWithResnet50Encoder, UNetWithResnet50Decoder
from online_triplet_loss.losses import *
from utils import get_accuracy
# from torchnlp.word_to_vector import FastText
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from gensim.models import fasttext
# from gensim.test.utils import datapath


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_weights(m_in):
    if type(m_in) == nn.Linear:
        torch.nn.init.kaiming_normal_(m_in.weight)
        m_in.bias.data.fill_(0.01)


def make_fc_1d(f_in, f_out):
    """
    Create a 1d fc layer
    """
    return nn.Sequential(nn.Linear(f_in, f_out),
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
	def __init__(self, args, num_classes_per_task, save_dir=None):
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
		if not save_dir:		
			self.word2vec = Word2Vec(args)
		self.word2vec = Word2Vec(args)
		self.visual2code = EmbedBranch(args.image_feat_dim, embedding_dim, metric_dim)
		self.code2word = EmbedBranch(metric_dim, embedding_dim, metric_dim)
		self.visual2word_mse = nn.MSELoss()
		self.classifier = Classifier(int(metric_dim*2), args.num_base_classes)	
		self.target_classifier = Classifier(int(metric_dim*2),args.num_base_classes)
		#self.reconstruction_loss = nn.MSELoss()
		self.save_dir = save_dir
		if self.args.dataset =='miniimagenet':
			self.d = {}
			with open("miniimagenet_trainsplit.txt") as f:
    				for line in f:
       					(key, val) = line.split()
       					self.d[str(key)] = str(val)
		if save_dir:
			checkpoint = torch.load(save_dir)
			self.decoder.load_state_dict(checkpoint['decoder'])
			self.text_branch.load_state_dict(checkpoint['text_branch'])
			self.image_branch.load_state_dict(checkpoint['image_branch'])
			self.visual2code.load_state_dict(checkpoint['visual2code'])
			self.code2word.load_state_dict(checkpoint['code2word'])
			self.classifier.load_state_dict(checkpoint['base_classifier'])
			self.target_classifier.load_state_dict(checkpoint['base_classifier'])
			
			for param in self.classifier.parameters():
            			param.requires_grad=False
			for param in self.text_branch.parameters():
            			param.requires_grad=False
			for param in self.image_branch.parameters():
            			param.requires_grad=False
			for param in self.visual2code.parameters():
            			param.requires_grad=False
			for param in self.code2word.parameters():
            			param.requires_grad=False
			for param in self.target_classifier.parameters():
            			param.requires_grad=False
			
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
	def target_save(self, save_path):
		torch.save({
				
				'decoder':self.decoder.state_dict(),
				'text_branch':self.text_branch.state_dict(),
				'image_branch':self.image_branch.state_dict(),
				'visual2code':self.visual2code.state_dict(),
				'code2word' : self.code2word.state_dict(),
				'base_classifier':self.classifier.state_dict(),
				'target_classifier':self.target_classifier.state_dict()
			
				}, save_path)


	def forward(self, batch):
		##training phase (support)
		save_dir = self.save_dir
		train_inputs, train_targets = batch['train']
		train_inputs = train_inputs.cuda()
		train_targets, train_original_labels = train_targets
		train_original_labels = list(zip(*train_original_labels))
		train_targets = train_targets.cuda()
		#print(np.shape(train_original_labels))
		#print(train_original_labels[0])#(np.transpose(np.array(train_original_labels))[:,1,:], train_targets)
		if self.args.dataset=='cifar_fs'and not save_dir:
			train_vecs = self.word2vec(np.transpose(np.array(train_original_labels)[:,1,:])).cuda()
		elif self.args.dataset=='miniimagenet'and not save_dir:
			train_vecs = [self.d[code] for code in np.array(train_original_labels[0])]
			#print(np.shape(np.array(train_vecs)), np.array(train_vecs))
			train_vecs = self.word2vec(np.array(train_vecs)[np.newaxis]).cuda()


		train_visual_bridge, pre_pools  = self.encoder(train_inputs.view(-1, *train_inputs.shape[2:])) #modification required to pass concatenated features through the decoder (separate encoder decoder structure maybe needed)
		#print(train_vecs.shape)
		train_visual_bridge  = torch.reshape(train_visual_bridge,(train_visual_bridge.size(0), train_visual_bridge.size(1)))#train_visual_bridge.permute(0,2,1,3).squeeze(3)#self.flatten(train_visual_bridge).unsqueeze(1)
		#print(train_visual_bridge.shape)
		"""
		if not save_dir:
			train_triplet_target = torch.cat((train_targets, train_targets))
			#train_targets = train_targets.cuda()
			train_triplet_target = torch.reshape(train_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_shots*2,1)).cuda()

			train_embed_visual = self.image_branch(train_visual_bridge)
			train_embed_text = self.text_branch(train_vecs)	#potential err yle loss calc due2 list type vecs 
			train_triplet_input = torch.cat((train_embed_visual, train_embed_text))
			#train split triplet loss
			#print(train_triplet_input.is_cuda, train_triplet_target.is_cuda)
			train_triplet_loss = self.triplet_loss(train_triplet_input.cuda(), train_triplet_target.cuda())

			#print('shape',np.shape(train_embed_visual),np.shape(train_embed_text))
			#concat visual and semantic features
			visual2code_out_train = self.visual2code(train_visual_bridge)
			visual2word_out_train = self.code2word(visual2code_out_train)
			train_concat_features = torch.cat((train_embed_visual, visual2word_out_train), dim=1) 
			#print('concat_dim', np.shape(train_concat_features))
	

		##testing phase (query)
		"""
		test_inputs, test_targets = batch['test']
		test_inputs = test_inputs.to(self.args.device)
		test_targets, test_original_labels = test_targets
		#print(np.shape(test_targets))
		test_original_labels = list(zip(*test_original_labels))
		test_targets = test_targets.to(self.args.device)

		if self.args.dataset=='cifar_fs' and not save_dir:
			test_vecs = self.word2vec(np.transpose(np.array(test_original_labels)[:,1,:])).cuda()
		elif self.args.dataset=='miniimagenet'and not save_dir:
			test_vecs = [self.d[code] for code in np.array(test_original_labels[0])]
			test_vecs = self.word2vec(np.array(test_vecs)[np.newaxis]).cuda()
		#test_vecs = self.word2vec(np.transpose(np.array(test_original_labels)[:,1,:])).cuda()
		test_visual_bridge, test_pre_pools = self.encoder(test_inputs.view(-1, *test_inputs.shape[2:])) #modification required to pass concatenated features through the decoder (separate encoder decoder structure maybe needed)
		test_visual_bridge  = torch.reshape(test_visual_bridge,(test_visual_bridge.size(0), test_visual_bridge.size(1)))
		"""
		#print(train_visual_bridge.shape)
		if not save_dir:
			test_triplet_target = torch.cat((test_targets, test_targets))
			test_triplet_target = test_triplet_target.cuda()
			test_triplet_target = torch.reshape(test_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_test_shots*2,1)).cuda()


			test_embed_visual = self.image_branch(test_visual_bridge)
			test_embed_text = self.text_branch(test_vecs)	#potential err yle loss calc due2 list type vecs 
			test_triplet_input = torch.cat((test_embed_visual, test_embed_text))

			#concat visual and semantic featurs
			test_concat_features = torch.cat((test_embed_visual, test_embed_text), dim=1) 


			#test split triplet loss
			test_triplet_loss = self.triplet_loss(test_triplet_input.cuda(), test_triplet_target.cuda())

		


			#reconstruct image
			train_decoder_out = self.decoder(train_concat_features.reshape(self.args.batch_size*self.args.num_shots*self.args.num_ways,self.args.image_feat_dim,1,1), pre_pools)
			test_decoder_out = self.decoder(test_concat_features.reshape(self.args.batch_size*self.args.num_test_shots*self.args.num_ways,self.args.image_feat_dim,1,1), test_pre_pools)
			#reconstruction loss
			#train_recon_loss = self.reconstruction_loss(train_decoder_out.unsqueeze(0),train_inputs)
			#test_recon_loss = self.reconstruction_loss(test_decoder_out.unsqueeze(0), test_inputs)
	
			#learn visual to word embedding
			visual2word_out_train = self.visual2word(train_visual_bridge)
			visual2word_loss_train = self.visual2word_mse(visual2word_out_train, train_embed_text)

			visual2word_out_test = self.visual2word(test_visual_bridge)
			visual2word_loss_test = self.visual2word_mse(visual2word_out_test, test_embed_text)

		"""
		#learn visual to word embedding
		#visual2word_out_train = self.visual2word(train_visual_bridge)
		visual2code_out_train = self.visual2code(train_visual_bridge)
		visual2word_out_train = self.code2word(visual2code_out_train)
		#train_concat_features = torch.cat((train_embed_visual, visual2word_out_train), dim=1) 
		
		train_vecs = self.word2vec(np.transpose(np.array(train_original_labels)[:,1,:])).cuda()
		test_vecs = self.word2vec(np.transpose(np.array(test_original_labels)[:,1,:])).cuda()
		test_embed_text = self.text_branch(test_vecs)	
		train_embed_text = self.text_branch(train_vecs)
		visual2word_loss_train = self.visual2word_mse(visual2word_out_train, train_embed_text)	
		#visual2word_out_test = self.visual2word(test_visual_bridge)
		visual2code_out_test = self.visual2code(test_visual_bridge)
		visual2word_out_test = self.code2word(visual2code_out_test) 
		visual2word_loss_test = self.visual2word_mse(visual2word_out_test, test_embed_text)
		train_embed_visual = self.image_branch(train_visual_bridge)
		test_embed_visual = self.image_branch(test_visual_bridge)
		train_concat_features_target = torch.cat((train_embed_visual, visual2word_out_train), dim=1) 
		test_concat_features_target = torch.cat((test_embed_visual, visual2word_out_test), dim=1)

		train_triplet_input = torch.cat((train_embed_visual, train_embed_text))
		train_triplet_target = torch.cat((train_targets, train_targets))
		#train_targets = train_targets.cuda()
		train_triplet_target = torch.reshape(train_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_shots*2,1)).cuda()

		#train split triplet loss
		#print(train_triplet_input.is_cuda, train_triplet_target.is_cuda)
		train_triplet_loss = self.triplet_loss(train_triplet_input.cuda(), train_triplet_target.cuda()) 

		test_triplet_input = torch.cat((test_embed_visual, test_embed_text))
		#train split triplet loss
		#print(train_triplet_input.is_cuda, train_triplet_target.is_cuda)
		test_triplet_target = torch.cat((test_targets, test_targets))
		#train_targets = train_targets.cuda()
		test_triplet_target = torch.reshape(test_triplet_target,(self.args.batch_size*self.args.num_ways*self.args.num_test_shots*2,1)).cuda()

		test_triplet_loss = self.triplet_loss(test_triplet_input.cuda(), test_triplet_target.cuda())
		#concatenated featurs thru FC layers
		if self.args.phase =='base':
			train_embeddings = self.classifier(train_concat_features)
			test_embeddings = self.classifier(test_concat_features)
		elif self.args.dataset=='cifar_fs':
			train_embeddings = self.target_classifier(train_concat_features_target)
			test_embeddings = self.target_classifier(test_concat_features_target)	
			base_train_embeddings = self.classifier(train_concat_features_target)
			base_test_embeddings = self.classifier(test_concat_features_target)	
	
			base_prototypes = get_prototypes(base_train_embeddings.unsqueeze(0), train_targets, self.num_classes_per_task)
			base_accuracy, base_logpy = get_accuracy(base_prototypes, base_test_embeddings.unsqueeze(0), test_targets)




		#fsl
		#print('prot',np.shape(test_embeddings.unsqueeze(0)),np.shape(test_targets), np.shape(self.num_classes_per_task))
		prototypes = get_prototypes(train_embeddings.unsqueeze(0), train_targets, self.num_classes_per_task)
		#print('proto',np.shape(prototypes))
		fsl_loss = prototypical_loss(prototypes, test_embeddings.unsqueeze(0), test_targets)
		

		##total loss
		if not save_dir:
			loss = fsl_loss + train_triplet_loss + test_triplet_loss + visual2word_loss_train + visual2word_loss_test #+ train_recon_loss + test_recon_loss
		elif self.args.phase=='test' or self.args.phase=='eval':
			loss = fsl_loss + train_triplet_loss + test_triplet_loss + visual2word_loss_train + visual2word_loss_test  
	

		##accuracy calculation
		accuracy, logpy = get_accuracy(prototypes, test_embeddings.unsqueeze(0), test_targets)
		#print(np.array(test_original_labels)[:,1,:])
		if not save_dir:
			return loss, accuracy, logpy
		elif self.args.phase=='test':
			if base_logpy > 10:
				return accuracy, loss, logpy, base_logpy
			else:
				return base_accuracy, loss, logpy, base_logpy
		else:
			if base_logpy > 10:
				return accuracy#loss, accuracy, logpy, base_logpy
			else:
				return base_accuracy#loss, base_accuracy, logpy, base_logpy


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

			
