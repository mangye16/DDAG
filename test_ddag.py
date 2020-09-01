from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_main import embed_net
from utils import *

import time 
import scipy.io as scio

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='Method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--graph', action='store_true', help='either add graph learning')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')
args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(1)
dataset = args.dataset
if dataset == 'sysu':
    # TODO: define your data path for RegDB dataset
    data_path = 'YOUR DATA PATH'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    # TODO: define your data path for RegDB dataset
    data_path = 'YOUR DATA PATH'
    n_class = 206
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa)
net.to(device)    
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    # model_path = checkpoint_path + 'test_best.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        # pdb.set_trace()
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}!!!!!!!!!!'.format(args.resume))


if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((280,150), interpolation=2),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)
    
    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    
    queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')
    
    gallset  = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = 2048
if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512

def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feature_dim))
    gall_feat_att = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, 0, test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr+batch_num,: ] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_att
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feature_dim))
    query_feat_att = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, 0, test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr+batch_num,: ] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat, query_feat_att
    
query_feat, query_feat_att = extract_query_feat(query_loader)

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
  
for trial in range(10):
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = trial)
    
    trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
    trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    
    gall_feat, gall_feat_att = extract_gall_feat(trial_gall_loader)
    
    # fc feature 
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP, mINP   = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
    
    # attention feature
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label,query_cam, gall_cam)
    if trial ==0:
        all_cmc = cmc
        all_mAP = mAP
        all_mINP = mINP
        all_cmc_att = cmc_att
        all_mAP_att = mAP_att
        all_mINP_att = mINP_att
    else:
        all_cmc = all_cmc + cmc
        all_mAP = all_mAP + mAP
        all_mINP = all_mINP + mINP
        all_cmc_att = all_cmc_att + cmc_att
        all_mAP_att = all_mAP_att + mAP_att
        all_mINP_att = all_mINP_att + mINP_att

    print('Test Trial: {}'.format(trial))
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC_att: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

cmc = all_cmc /10 
mAP = all_mAP /10
mINP = all_mINP /10

cmc_att = all_cmc_att /10
mAP_att = all_mAP_att /10
mINP_att = all_mINP_att /10
print ('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('FC_att: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
