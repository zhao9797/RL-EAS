import os
import numpy as np
import sys
import time
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from transformers import AdamW#, get_linear_schedule_with_warmup
from .funcs import Evaluation
def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class SumTrainer(object):

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, opts):

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.opts = opts
        self.eval = Evaluation(opts)

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(self, model):
        print("Start training...")

        # Init
        if self.opts.bert_optim:
            print('Use bert optim!')
            assert 0
            # parameters_to_optimize = list(model.named_parameters())
            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # parameters_to_optimize = [
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # ]

            # optimizer = AdamW(parameters_to_optimize, lr=self.opts.lr, correct_bias=False)
        else:
            if self.opts.optim == "sgd":
                pytorch_optim = optim.SGD
            elif self.opts.optim == "adam":
                pytorch_optim = optim.Adam
            optimizer = pytorch_optim(model.parameters(),self.opts.lr, weight_decay=self.opts.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.lr_step_size)

            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
            #                                             num_training_steps=train_iter)

        if self.opts.load_ckpt:
            state_dict = self.__load_model__(self.opts.load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_rouge = 0
        not_best_count = 0  # Stop training after several epochs without improvement.

        for epoch in range(self.opts.epochs):
            avg_loss = 0.0
            self.eval.reset()
            for i, data in enumerate(self.train_data_loader):
                document  = data[0]
                label = data[1]
                if self.opts.cuda:
                    for k in document:
                        document[k] = document[k].cuda()
                        label[k] = label[k].cuda()
                #### forword
                logits = model(document)
                loss = model.loss(logits, label)
                pred = []
                for p in logits:
                    idx = np.where(p.cpu()>0.5)[0]
                    idx = idx.tolist()
                    pred.append(idx)
                p, r, f = self.eval.per_eval(pred, label)
                loss.backward()  # retain_graph=True
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss += loss.data.item()
                sys.stdout.write(
                    'epoch:{0:4} | step: {1:4} | loss: {2:2.6f}, p: {3:3.2f}%, r: {4:3.2f}%, f1-score: {5:3.2f}%'
                    .format(epoch, i + 1, avg_loss / (i+1), p, r, f) + '\r')
                sys.stdout.flush()

            precision, recall, f1_score, rouge = self.eval(model, self.val_data_loader)
            model.train()
            if rouge > best_rouge:
                print('Best checkpoint')
                torch.save({'state_dict': model.state_dict()}, self.opts.save_ckpt)
                best_rouge = rouge
        print("\n####################\n")
        print("Finish training "+self.opts.model_name)

    def eval(self, model, data_loader):
        print("")
        model.eval()
        all_logits = []
        with torch.no_grad():
            for it, data in enumerate(data_loader):
                document  = data[0]
                label = data[1]
                if self.opts.cuda:
                    for k in document:
                        document[k] = document[k].cuda()
                    label = label.cuda()
                logits, pred = model(document)
                for i in range(logits.size(0)):
                    all_logits.append(logits[i].item())

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / it+1) + '\r')
                sys.stdout.flush()
            print("")
        self.eval.reset()
        precision, recall, f1_score, r = self.eval.evaluation(all_logits, data_loader.dataset)
        return precision, recall, f1_score, r
