import argparse
import os
import logging
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PromptDataset
from temping import get_temps
from model import Model

import transformers

from transformers import (
                AdamW,
                RobertaConfig,
                RobertaTokenizer,
                get_scheduler,
                set_seed,
                get_linear_schedule_with_warmup
                )

def parse_args():
    parser = argparse.ArgumentParser("Classification using prompt")
    parser.add_argument("--from_pretrained", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path of the training and validation data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument('--epoch', type=int, default=5,
                        help='num of epochs')
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Weight decay rating")
    parser.add_argument("--num_of_epochs", type=int, default=10,
                        help="number of train epochs")
    parser.add_argument("--max_steps", type=int, required=True,
                        help="max_train_steps, equal EPOCH * NUM_SAMPLES / BATCH_SIZE")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="max input sentence length")
    parser.add_argument("--save_model_dir", type=str, required=True,
                        help="directory to save the trained model")
    parser.add_argument("--seed", type=int, default=None, 
                        help="A seed for reproducible training.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    args = parser.parse_args()

    return args

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` 
    Args:
    seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.from_pretrained)
    
    temps = get_temps(args, tokenizer)

    train_dataset = PromptDataset(args.data_dir, 'train', tokenizer = tokenizer, temps = temps, max_length = args.max_seq_len)

    dev_dataset = PromptDataset(args.data_dir, 'dev', tokenizer = tokenizer, temps = temps, max_length = args.max_seq_len)

    test_dataset = PromptDataset(args.data_dir, 'test', tokenizer = tokenizer, temps = temps, max_length = args.max_seq_len)

    train_loader = DataLoader(train_dataset, 
                            batch_size = args.batch_size,
                            shuffle=True, 
                            num_workers=8)
    dev_loader = DataLoader(dev_dataset, 
                            batch_size = args.batch_size,
                            shuffle=False,
                            num_workers=8)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8)

    model = Model(args)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay':args.wd,
                },
                {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay':0.0},]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = int(args.max_steps * 0.1),
                num_training_steps = args.max_steps)

    loss_fc = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(args.epoch):
        for step, d in enumerate(train_loader):
            input_ids, attention_mask, lm_labels, locs = d
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            locs = locs.squeeze(1).numpy()

            logits = model(input_ids, attention_mask, locs, lm_labels.squeeze(1).numpy())
            lm_labels = lm_labels.squeeze(1).to(device)
            logits = torch.cat(logits, dim = 0)
            loss = loss_fc(logits, lm_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step % 50 == 0:
                _lr = optimizer.state_dict()['param_groups'][0]['lr']
                _loss = loss.item()
                msg = '[epoch-%d-step-%d] train loss %.5f lr %.3e' % (epoch, step, _loss, _lr)
                logging.info(msg)

            if step % 300 == 0:
                acc = []
                with torch.no_grad():
                    model.eval()
                    for e_step, d in enumerate(dev_loader):
                        input_ids, attention_mask, lm_labels, locs = d
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        locs = locs.squeeze(1).numpy()
                        logits = model(input_ids, attention_mask, locs, lm_labels.squeeze(1).numpy())
                        lm_labels = lm_labels.squeeze(1).to(device)
                        logits = torch.cat(logits, dim = 0)
                        a = logits.argmax(dim = -1)
                        a = a == lm_labels
                        acc.append(a.cpu().numpy())

                acc = np.concatenate(acc).mean()
                logging.info('dev acc %.5f' % acc)

                test_acc = []
                with torch.no_grad():
                    for e_step, d in enumerate(test_loader):
                        input_ids, attention_mask, lm_labels, locs = d
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        locs = locs.squeeze(1).numpy()
                        logits = model(input_ids, attention_mask, locs, lm_labels.squeeze(1).numpy())
                        lm_labels = lm_labels.squeeze(1).to(device)
                        logits = torch.cat(logits, dim = 0)
                        a = logits.argmax(dim = -1)
                        a = a == lm_labels
                        test_acc.append(a.cpu().numpy())
                model.train()
                test_acc = np.concatenate(test_acc).mean()
                logging.info('test acc %.5f' % test_acc)

                torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'epoch_{}_step_{}_acc_{:.5f}ckpt.pt'.format(epoch, step, test_acc)))

if __name__=="__main__":
    main()
