import os
import json
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer, BertTokenizerFast

squad_path = './data/squad/v2.0'
BATCH_SIZE = 8

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

def train(gpu, args):
    args.gpu = gpu
    # rank calculation for each process per gpu so that they can be identified uniquely.
    rank = args.local_ranks * args.ngpus + gpu
    print('rank:',rank)


    # set the gpu for each processes
    torch.cuda.set_device(args.gpu)

    train_encodings, val_encodings = get_squad_encodings(squad_path)
    train_dataset = SquadDataset(train_encodings)
    val_dataset   = SquadDataset(val_encodings)

    # Ensures that each process gets differnt data from the batch.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = (train_sampler is None),
        num_workers = 4,
        sampler     = train_sampler,
    )

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # Boilerplate code to initialize the parallel prccess.
    # It looks for ip-address and port which we have set as environ variable.
    # If you don't want to set it in the main then you can pass it by replacing
    # the init_method as ='tcp://<ip-address>:<port>' after the backend.
    # More useful information can be found in
    # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    # start from the same randomness in different nodes. If you don't set it
    # then networks can have different weights in different nodes when the
    # training starts. We want exact copy of same network in all the nodes.
    # Then it will progress from there.

    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        #"bert-large-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    )
    model = model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    # create an optimizer object
    optim = AdamW(model.parameters(), lr=5e-5)


    for epoch in range(3):
        print('epoch ', epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, \
                 start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

    #model.eval()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nodes',       default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int, help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ngpus',       default=1, type=int, help='number of gpus per node')

    args = parser.parse_args()

    args.world_size           = args.ngpus * args.nodes
    os.environ['WORLD_SIZE']  = str(args.world_size)
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    #mp.spawn(train, nprocs=args.ngpus, args=(args,))
    mp.spawn(train, nprocs=args.ngpus, args=(args,), join=True)
