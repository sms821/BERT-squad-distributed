import os
import sys
import json
import time
import math
from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser

import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


from transformers import AdamW
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer, BertTokenizerFast

sys.path.append('/home/cc/DS-Analyzer/tool')
from profiler_utils import DataStallProfiler

squad_path = '/home/cc/BERT-squad-distributed/examples/pytorch/data/squad/v2.0'

parser = argparse.ArgumentParser(description='PyTorch BERT')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')

parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
        help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--noeval', action='store_true')
parser.add_argument('--amp',action='store_true',help='Run model AMP (automatic mixed precision) mode.')
parser.add_argument("--nnodes", default=1, type=int)
parser.add_argument("--node_rank", default=0, type=int)
parser.add_argument("--delay_allreduce", default=True, type=bool)
parser.add_argument("--data", default='', type=str)

#profiler
parser.add_argument('--data-profile', action='store_true', default=True,
         help='Set profiler on')
parser.add_argument('--synthetic', action='store_true',
         help='Use synthetic dataset')
parser.add_argument('--suffix', type=str, default=None,
         help='Suffix for the data logs during profiling')
parser.add_argument('--tensor_path', type=str, default=None)
parser.add_argument('--num_minibatches', type=int, default=50)
parser.add_argument("--precreate", action='store_true',
                        help="Precreated tensors loaded from file")
parser.add_argument("--full_epoch", action='store_true', default=False)
parser.add_argument("--classes", default=1000, type=int)
parser.add_argument("--arch", default=None, type=str)


cudnn.benchmark = True

args = parser.parse_args()

#profile mode
#redirect all output to a log file
if args.data_profile:
    args.dprof = DataStallProfiler(args)

BATCH_SIZE = args.batch_size
#args.world_size    = args.nnodes
#os.environ['WORLD_SIZE']  = str(args.world_size)
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '56070'

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

cudnn.benchmark = True

compute_time_list = []
data_time_list = []
fwd_prop_time_list = []
bwd_prop_time_list = []


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers



def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def get_squad_encodings(path):
    train_contexts, train_questions, train_answers = \
        read_squad(os.path.join(squad_path, 'train-v2.0.json'))
    val_contexts, val_questions, val_answers = \
        read_squad(os.path.join(squad_path, 'dev-v2.0.json'))

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased', \
            do_lower_case=True)

    train_encodings = tokenizer(train_contexts, train_questions, \
            truncation=True, padding=True)
    val_encodings   = tokenizer(val_contexts, val_questions, truncation=True, \
            padding=True)

    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)

    return train_encodings, val_encodings


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

#print('num batches ', len(train_loader))

def train(train_loader, model, optim, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    args.dprof.start_data_tick()
    dataset_time = compute_time = 0
    device   = args.gpu
    
    for i, batch in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        dataset_time += (time.time() - end)
        compute_start = time.time()


        #-----------------Stop data, start compute------#
        #if profiling, sync here
        if args.data_profile:
            torch.cuda.synchronize()
            args.dprof.stop_data_tick()
            args.dprof.start_compute_tick()
        #-----------------------------------------------#

        outputs = model(input_ids, attention_mask=attention_mask, \
             start_positions=start_positions, end_positions=end_positions)

        loss = outputs[0]

        # compute gradient and do SGD step
        args.dprof.start_compute_bwd_tick()

        loss.backward()

        args.dprof.start_AR_tick()
        optim.step()
        args.dprof.stop_AR_tick()

        torch.cuda.synchronize()

        #-----------------Stop compute, start data------#
        args.dprof.stop_compute_bwd_tick()
        args.dprof.stop_compute_tick()
        args.dprof.start_data_tick()
        #-----------------------------------------------#

        compute_time += (time.time() - compute_start)

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        #train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
        train_loader_len = int(math.ceil(len(train_loader) / args.batch_size))

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)

    return batch_time.avg

    #model.eval()

def main():
    start_full = time.time()
    global args

    time_stat = []
    start = time.time()

    args.gpu = 0
    args.world_size = 1
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    # rank calculation for each process per gpu so that they can be identified uniquely.
    #rank = args.local_ranks * args.ngpus + gpu
    rank = args.local_rank
    print('rank:', args.local_rank)

    train_encodings, val_encodings = get_squad_encodings(squad_path)
    train_dataset = SquadDataset(train_encodings)
    val_dataset   = SquadDataset(val_encodings)

    # Ensures that each process gets differnt data from the batch.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.local_rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = (train_sampler is None),
        num_workers = 0,
        sampler     = train_sampler,
    )

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '56070'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.manual_seed(0)
    # start from the same randomness in different nodes. If you don't set it
    # then networks can have different weights in different nodes when the
    # training starts. We want exact copy of same network in all the nodes.
    # Then it will progress from there.

    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    )
    model = model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    # create an optimizer object
    optim = AdamW(model.parameters(), lr=5e-5)

    total_time = AverageMeter()
    dur_setup = time.time() - start
    time_stat.append(dur_setup)
    print("Batch size for GPU {} is {}, workers={}".format(args.gpu, args.batch_size, args.workers))

    for epoch in range(args.start_epoch, args.epochs):
        print('epoch ', epoch)

        # log timing
        start_ep = time.time()

        avg_train_time = train(train_loader, model, optim, epoch)

        total_time.update(avg_train_time)

        dur_ep = time.time() - start_ep
        print("EPOCH DURATION = {}".format(dur_ep))
        time_stat.append(dur_ep)


    if args.local_rank == 0:
        for i in time_stat:
            print("Time_stat : {}".format(i))

        for i in range(0, len(data_time_list)):
            print("Data time : {}\t Compute time : {}".format(data_time_list[i], compute_time_list[i]))

    dur_full = time.time() - start_full

    if args.local_rank == 0:
        print("Total time for all epochs = {}".format(dur_full))

    if args.data_profile:
        args.dprof.stop_profiler()


if __name__ == '__main__':

    main()

