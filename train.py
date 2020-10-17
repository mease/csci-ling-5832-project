# Code adapted from the follwoing references:
# https://lionbridge.ai/articles/transformers-in-nlp-creating-a-translator-model-from-scratch/
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dca13261bbb4e9809d1a3aa521d22dd7/transformer_tutorial.ipynb

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import logging

import argparse
import torchtext.data as ttdata
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time

from model import Transformer
from tokenize_methods import tokenize_nltk


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"


def greedy_decode_sentence(model, sentence, SRC, TGT):
    model.eval()
    sentence = SRC.preprocess(sentence)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 :
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    sentence = Variable(torch.LongTensor([indexed])).cuda()
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).cuda()
    translated_sentence = ""
    maxlen = 50
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda()
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence += " " + add_word
        if add_word == EOS_WORD:
            break
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
    return translated_sentence


def train_epoch(ds_iter, model, optim, batch_size, train=True, use_gpu=True):
    epoch_loss = 0
    
    for i, batch in enumerate(ds_iter):
            src = batch.src.cuda() if use_gpu else batch.src
            tgt = batch.tgt.cuda() if use_gpu else batch.tgt
            #change to shape (bs , max_seq_len)
            src = src.transpose(0,1)
            #change to shape (bs , max_seq_len+1) , Since right shifted
            tgt = tgt.transpose(0,1)
            tgt_input = tgt[:, :-1]
            targets = tgt[:, 1:].contiguous().view(-1)
            src_mask = (src != 0)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda() if use_gpu else src_mask
            tgt_mask = (tgt_input != 0)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            tgt_mask = tgt_mask.cuda() if use_gpu else tgt_mask
            size = tgt_input.size(1)
            np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask   
            # Forward, backprop, optimizer
            if (train):
                optim.zero_grad()
            preds = model(src.transpose(0,1), tgt_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=tgt_mask)
            preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
            if (train):
                loss.backward()
                optim.step()
            
            epoch_loss += loss.item() / batch_size
            
    return epoch_loss / len(ds_iter)


def train(train_iter, val_iter, model, optim, num_epochs, batch_size,
          test_src_sentence, test_tgt_sentence, SRC, TGT, use_gpu=True):
    train_losses = []
    valid_losses = []
    best_epoch = -1
    best_train_loss = 100000;
    best_valid_loss = 100000;
    train_start = time.time()
    for epoch in range(num_epochs):
        logging.info(f'''Starting Epoch [{epoch+1}/{num_epochs}]''')
        epoch_start = time.time()
        
        # Train model
        model.train()
        train_loss = train_epoch(train_iter, model, optim, batch_size, True)
        
        model.eval()
        with torch.no_grad():
            valid_loss = train_epoch(val_iter, model, optim, 1, False)
        
        epoch_time = time.time() - epoch_start
        
        logging.info(f'''Epoch [{epoch+1}/{num_epochs}] complete in {epoch_time:.3f} seconds.''')
        logging.info(f'''Train Loss: {train_loss:.3f}. Val Loss: {valid_loss:.3f}''')
        
        if valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            
            logging.info("Saving state dict")
            torch.save(model.state_dict(), 'checkpoint_best_epoch.pt')
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Check Example after each epoch:
        logging.info('')
        logging.info(f"Source: {test_src_sentence}")
        logging.info('')
        logging.info(f"Target: {test_tgt_sentence}")
        logging.info('')
        logging.info(f"Predicted: {greedy_decode_sentence(model, test_src_sentence, SRC, TGT)}")
        logging.info("-----------------------------------------")
        logging.info('')
    
    train_time = time.time() - train_start
    logging.info(f'''Training complete in {train_time/60/60:.3f} hours.''')
    logging.info(f'''Best train loss: {best_train_loss}. Best val loss: {best_valid_loss}. Attained at epoch {best_epoch}''')
    return train_losses, valid_losses


def main(tok_method, train_file, val_file, num_epochs, batch_size, learning_rate, data_path):
    SRC = ttdata.Field(tokenize=tok_method, pad_token=BLANK_WORD)
    TGT = ttdata.Field(tokenize=tok_method, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
    
    logging.info('Loding training data...')
    train_ds, val_ds = ttdata.TabularDataset.splits(
        path=data_path, format='tsv',
        train=train_file,
        validation=val_file,
        fields=[('src', SRC), ('tgt', TGT)]
    )
    
    test_src_sentence = ' '.join(val_ds[0].src)
    test_tgt_sentence = ' '.join(val_ds[0].tgt)
    
    MIN_FREQ = 2
    SRC.build_vocab(train_ds.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_ds.tgt, min_freq=MIN_FREQ)
    
    logging.info(f'''SRC vocab size: {len(SRC.vocab)}''')
    logging.info(f'''TGT vocab size: {len(TGT.vocab)}''')
    
    train_iter = ttdata.BucketIterator(train_ds, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))
    val_iter = ttdata.BucketIterator(val_ds, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))
    
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)
    
    model = Transformer(source_vocab_length=source_vocab_length,target_vocab_length=target_vocab_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    model = model.cuda()
    
    train_losses,valid_losses = train(train_iter, val_iter,
                                      model, optim, num_epochs, batch_size,
                                      test_src_sentence, test_tgt_sentence,
                                      SRC, TGT)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train_file', help='Source language file.')
    parser.add_argument('val_file', help='Target language file.')
    parser.add_argument('num_epochs', help='Number of epochs to train.')
    parser.add_argument('batch_size', help='Batch size.')
    parser.add_argument(
        '--data_path',
        default='data',
        help='Directory containing the training files.')
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        help='Adam optimizer learning rate.')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.DEBUG,
                        filename='train.log',
                        format='%(message)s',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    
    main(tokenize_nltk,
         args.train_file,
         args.val_file,
         int(args.num_epochs),
         int(args.batch_size),
         float(args.learning_rate),
         args.data_path)
    