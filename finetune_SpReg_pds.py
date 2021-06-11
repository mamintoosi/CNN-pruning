import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import matplotlib.pyplot as plt
from sparse_regularization import sparse_regularization
from torchvision import datasets, transforms
import random
import cPickle as pkl

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v.cpu() / np.sqrt(torch.sum(v.cpu() * v.cpu())) #M.Amintoosi
            self.filter_ranks[i] = v#.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner_VGG16:
    def __init__(self, ds_name, model):
        # self.train_data_loader = dataset.loader(train_path)
        # self.test_data_loader = dataset.test_loader(test_path)
        # self.eval_data_loader = dataset.eval_loader(test_path,batch_size=1,num_workers=0)
        if ds_name == 'CIFAR10':
            trainset = datasets.CIFAR10(root=data_path,train=True,download=True,transform=transform_train)
            testset = datasets.CIFAR10(root=data_path,train=False,download=True,transform=transform_test)
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            frac = 0.1
        if ds_name == 'FashionMNIST':
            trainset = datasets.FashionMNIST(root=data_path,train=True,download=True,transform=transform_train)
            testset = datasets.FashionMNIST(root=data_path,train=False,download=True,transform=transform_test)
            classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
            frac = 0.1
        elif ds_name == 'STL10':
            trainset = datasets.STL10(root=data_path,split='train',download=True,transform=transform_train)
            testset = datasets.STL10(root=data_path,split='test',download=True,transform=transform_test)
            classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
            frac = 0.5
            
        batch_size = 32
        workers = 2

        # انتخاب زیرمجموعه‌ای کوچک‌تر برای سرعت بیشتر
        List = range(len(trainset))
        rnd_subset = random.sample(List, int(len(trainset)*frac)) 
        trainset_sub = torch.utils.data.Subset(trainset, rnd_subset)
        # print('Len TrainSet',len(trainset), len(trainset_sub))
        List = range(len(testset))
        rnd_subset = random.sample(List, int(len(testset)*frac)) 
        testset_sub = torch.utils.data.Subset(testset, rnd_subset)
    
        trainloader = torch.utils.data.DataLoader(trainset_sub,batch_size=batch_size,shuffle=True,num_workers=workers)
        testloader = torch.utils.data.DataLoader(testset_sub,batch_size=batch_size,shuffle=False, num_workers=workers)

        self.train_data_loader = trainloader
        self.test_data_loader = testloader
        self.eval_data_loader = testloader

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model) 
        self.model.train()

    def test(self):
        # return
        # print("Test starts...")
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()

    def eval_test_results(self):
        class_names = ['COVID','Normal']
        plt.figure(figsize=(25,10))
        count = np.zeros(2)

        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.eval_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            # print(pred.cpu().eq(label).sum())
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            isPredCorrect = pred.cpu().eq(label)
            total += label.size(0)
            if count[label]>=5:
                continue
            count[label]+=1
            plt.subplot(2,5,5*label+count[label])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            inp = np.transpose(batch[0].numpy(), (1, 2, 0))#inp.numpy().transpose((1, 2, 0))
            mean = [0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if isPredCorrect==True:
                color = 'blue'
            else:
                color = 'red'
            plt.xlabel("{}".format(class_names[pred]),color=color, fontsize=25)
        plt.show()        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()

    def train(self, optimizer = None, epoches=10, regularization=None):
        if optimizer is None:
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
            # optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i+1, '/', epoches)
            self.train_epoch(optimizer,regularization=regularization)
            self.test()
        print("Finished fine tuning.")
        

    def train_epoch(self, optimizer = None, rank_filters = False, regularization = None):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters, regularization)

    def train_batch(self, optimizer, batch, label, rank_filters, regularization=None):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            loss = self.criterion(output, Variable(label))
            loss.backward()
        else:
            loss = self.criterion(self.model(input), Variable(label))
            if regularization is not None:
                # print('Using Regularization: ',reg_name)
                loss += 1e-8*regularization(0.5)
            loss.backward()
            optimizer.step()

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        # print(number_of_filters, num_filters_to_prune_per_iteration, iterations)
        iterations = int(iterations * 7. / 10) 
        print(iterations)
        print("Number of prunning iterations to reduce 70% filters", iterations)

        dics = [] # A list for saving dics
        for i in range(iterations):
            print("Iter: ", i+1, '/', iterations)
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)

            # Add to list for future saving
            dics.append(layers_prunned)

            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100 - 100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            if args.eat == 'dec':
                self.train(optimizer, epoches=iterations-i)
            else:
                self.train(optimizer, epoches=int(args.eat))


        # print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epoches=5) 
        # models_dir = 'models/'
        # torch.save(model.state_dict(), models_dir+"painting_model_prunned.pt")
        # torch.save(model, models_dir+"VGG_model_COVID19_prunned.pt")
        # model_file_name = '{}_prnIn_{}_reg-{}_pruned.pt'.format(args.models_dir, \
        #     args.prune_input,args.ds_name, args.reg_name)
        model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        torch.save(model, model_file_name)

        dic_file_name = '{}.pkl'.format(args.output_model)
        pkl.dump(dics, open(dic_file_name, "wb" ) )


    def prune_reg(self):
        if args.reg_name is not None:
            device = torch.device("cuda" if args.use_cuda else "cpu") #
            regularization = sparse_regularization(self.model,device)
            # reg_name = 'HSQGL12'
            if reg_name == 'L2':
                regularizationFun = regularization.l2_regularization
            elif reg_name == 'L1':
                regularizationFun = regularization.l1_regularization
            elif reg_name == 'HSQGL12':
                regularizationFun = regularization.hierarchical_squared_group_l12_regularization
            print('Using Regularization: ',reg_name)
        else:
            regularizationFun = None

        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True
        print("Retraining with regularization ... ")
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train(optimizer, epoches = args.train_epoch, regularization = regularizationFun)

        # model_file_name = '{}_prnIn-{}_{}_reg-{}.pt'.format(args.models_dir, \
        #     args.prune_input,args.ds_name, args.reg_name)
        model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        torch.save(model, model_file_name)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--train_path", type = str, default = "data/train")
    parser.add_argument("--test_path", type = str, default = "data/test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')    
    parser.add_argument('--reg_name', type = str, default = None)
    parser.add_argument('--ds_name', type = str, default = 'COVID-CT')
    parser.add_argument("--train_epoch", type = int, default = 15)
    parser.add_argument('--prune_input', type = str, default = 'vgg')
    parser.add_argument('--input_model', type = str, default = 'vgg')
    parser.add_argument('--output_model', type = str, default = 'taylor')
    # Epochs After tuning: constant or decremental, const, dec
    parser.add_argument('--eat', type = str, default = '5') 
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    # global args 
    args = get_args()

    # dsName = args.ds_name #'cifar10'
    num_classes = 10

    data_path = '/content/data'
    transform_test = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
            ])
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py            
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),            

# if dataset == 'cifar10':
    transform_train = transforms.Compose([
            # transforms.RandomCrop(32,padding = 4),
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])


    args.models_dir = 'models/'
    reg_name = args.reg_name

    if args.train:
        model = ModifiedVGG16Model()
    elif args.test or args.prune:
        model_file_name = '{}{}.pt'.format(args.models_dir, args.input_model)
        model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
        # pasvand = ''
        # if args.reg_name is not None:
        #     pasvand = '_reg-{}'.format(args.reg_name)
        # if args.prune:
        #     pasvand += '_pruned'
        # model_file_name = '{}VGG_model_{}{}.pt'.format(args.models_dir, \
        #     args.ds_name, pasvand)
        # model = torch.load(models_dir+"VGG_model_COVID19_prunned.pt", map_location=lambda storage, loc: storage)
        # model = torch.load(models_dir+"painting_model_reg_prunned.pt", map_location=lambda storage, loc: storage)
    # elif args.prune:
    #     # if args.prune_input is None or args.prune_input == 'vgg':
    #     #     model_file_name = '{}VGG_model_{}.pt'.format(args.models_dir, \
    #     #     args.ds_name)
    #     # elif args.prune_input == 'taylor':
    #     #     model_file_name = '{}VGG_model_{}_reg-{}_pruned.pt'.format(args.models_dir, \
    #     #     args.ds_name,args.reg_name)
    #     # elif args.prune_input == 'reg':
    #     #     model_file_name = '{}VGG_model_{}_reg-{}.pt'.format(args.models_dir, \
    #     #     args.ds_name,args.reg_name)
    #     model_file_name = '{}{}.pt'.format(args.models_dir,args.input_model)
    #     model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    #     # model = torch.load(models_dir+"VGG_model_COVID19.pt", map_location=lambda storage, loc: storage)
    
    if args.use_cuda:
        model = model.cuda()
        print('Using CUDA...')
    
    fine_tuner = PrunningFineTuner_VGG16(args.ds_name, model)

    if args.train:            
        fine_tuner.train(epoches=args.train_epoch)#, regularization=regularizationFun)
        model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        # model_file_name = '{}VGG_model_{}.pt'.format(args.models_dir, \
        #     args.ds_name)
        torch.save(model, model_file_name)
    elif args.prune:
        if args.reg_name is None:
            fine_tuner.prune()
        else:
            fine_tuner.prune_reg()

    elif args.test:
        fine_tuner.eval_test_results()

    # print(model)
