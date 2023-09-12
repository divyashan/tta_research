import os
import torch

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

from utils import accuracy, AverageMeter, ProgressMeter
from tta_models import AugTTA, ClassTTA
from paths import AGG_MODELS_DIR

# TODO port over imagenet utils from old repo

from tta_models import ClassTTA, AugTTA

def train_tta_model(expt_config, examples, target):
    # expt config must contain model_name, aug_name, epochs, agg_name, dataset, n_classes, temp_scale=1, initialization='even'
    model_name = expt_config['model_name']
    aug_policy = expt_config['aug_policy']
    agg_name = expt_config['agg_name']
    epochs = expt_config['epochs']
    dataset = expt_config['dataset']
    n_classes = expt_config['n_classes']
    temp_scale = expt_config['temp_scale']
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    n_augs = len(aug_policy.split(','))
    criterion = torch.nn.CrossEntropyLoss()
    if agg_name == 'AugTTA':
        model = AugTTA(n_augs,n_classes,temp_scale)
    elif agg_name == 'ClassTTA':
        model = ClassTTA(n_augs,n_classes,temp_scale)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)

    model.cuda('cuda:0')
    criterion.cuda('cuda:0')
    model.train()
    
    lamda = .01 
    params = torch.cat([x.view(-1) for x in model.parameters()])

    
    examples = torch.Tensor(examples)
    target = torch.Tensor(target).long()

    losses_arr = []
    accs_arr = []
    for epoch in range(epochs):
        
        batch_size = 1000
        n_batches = int(len(examples)/batch_size + 1)
        progress = ProgressMeter(n_batches,
                        [batch_time, data_time, losses, top1, top5],
                        prefix="Epoch: [{}]".format(epoch))
        
        loss_vals = []
        acc_vals = []
        for i in range(n_batches):
            example_batch = examples[i*batch_size:(i+1)*batch_size]
            target_batch = target[i*batch_size:(i+1)*batch_size]
            if len(target_batch) == 0:
                continue
            example_batch = example_batch.cuda('cuda:0', non_blocking=True)
            target_batch = target_batch.cuda('cuda:0', non_blocking=True)
            output = model(example_batch)
            nll_loss = criterion(output, target_batch)
            l1_loss = lamda * torch.norm(params, 1)
            loss = nll_loss 
            acc1, acc5 = accuracy(output, target_batch, topk=(1,5))

            losses.update(loss.item(), examples.size(0))
            top1.update(acc1[0], examples.size(0))
            top5.update(acc5[0], examples.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            for p in model.parameters():
                p.data.clamp_(0)                 
            loss_vals.append(loss.item())
            acc_vals.append(acc1[0].item())
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
#         progress.display(epoch)
    
    
    model_prefix = AGG_MODELS_DIR + '/' + model_name + '/' + aug_policy
    os.makedirs(model_prefix, exist_ok=True)
    torch.save(model.state_dict(), model_prefix + '/' + agg_name + '_lr.pth')
    
    acc_logs_path = AGG_MODELS_DIR + '/' + model_name + '/' + aug_policy + '/' + agg_name + '_accs'
    loss_logs_path = AGG_MODELS_DIR + '/' + model_name + '/' + aug_policy + '/' + agg_name + '_losses'
    
    np.savetxt(loss_logs_path, losses_arr)
    np.savetxt(acc_logs_path, accs_arr)
        
    return model