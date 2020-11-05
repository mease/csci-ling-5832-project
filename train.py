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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf

from model import Transformer
from tokenize_methods import TokenizerWrapper


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '[UNK]'
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
PAD_TOKEN = '[PAD]'
MASK_TOKEN = '[MASK]'

SPECIAL_TOKENS = [BOS_WORD, EOS_WORD, BLANK_WORD, SEP_TOKEN, CLS_TOKEN, PAD_TOKEN, MASK_TOKEN]


def greedy_decode_sentence(model, sentence, SRC, TGT, tokenizer):
    sentence = SRC.preprocess(sentence)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 :
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    return greedy_decode_ids(model, indexed, SRC, TGT, tokenizer)

def greedy_decode_ids(model, sentence, SRC, TGT, tokenizer):
    model.eval()
    sentence = Variable(torch.LongTensor([sentence])).cuda()
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).cuda()
    tokens = []
    if tokenizer.tok_type == 'char':
        maxlen = 500
    else:
        maxlen = 50
        
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda()
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        
        if add_word == EOS_WORD:
            break
        tokens.append(add_word)
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
    
    return tokenizer.decode(tokens, BLANK_WORD)

def score(ds_iter, model, tgt_tokenizer, SRC, TGT):
    bleu_tot = 0.0
    chrf_tot = 0.0
    count = 0
    model.eval()
    for i, batch in enumerate(ds_iter):
        src = batch.src.transpose(0,1)[0].numpy()
        tgt = batch.tgt.view(-1).numpy()
        tgt_tokens = []
        for index in tgt:
            tgt_tokens.append(TGT.vocab.itos[index])
        
        pred_sentence = greedy_decode_ids(model, src, SRC, TGT, tgt_tokenizer).strip().split(' ')
        tgt_sentence = tgt_tokenizer.decode(tgt_tokens[1:-1], BLANK_WORD).strip().split(' ')

        bleu_tot += sentence_bleu([tgt_sentence], pred_sentence)
        try:
            chrf_tot += sentence_chrf(tgt_sentence, pred_sentence)
        except:
            # Ignore
            chrf_tot += 0.0
        count += 1
    return bleu_tot/count, chrf_tot/count
        

def train_epoch(ds_iter, model, optim, batch_size, tgt_tokenizer, SRC, TGT, train=True, use_gpu=True):
    epoch_loss = 0
    #bleu_score = 0.0
    
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
            if train:
                optim.zero_grad()
            preds = model(src.transpose(0,1), tgt_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=tgt_mask)
            preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds, targets, ignore_index=0, reduction='sum')

            if train:
                loss.backward()
                optim.step()
            
            epoch_loss += loss.item() / batch_size
            
    return epoch_loss / len(ds_iter)


def train(train_iter, val_iter, model, optim, num_epochs, batch_size,
          test_src_sentence, test_tgt_sentence, SRC, TGT, src_tokenizer,
          tgt_tokenizer, checkpoint_file, use_gpu=True):
    train_losses = []
    valid_losses = []
    best_epoch = -1
    best_train_loss = 100000
    best_valid_loss = 100000

    train_start = time.time()
    for epoch in range(num_epochs):
        logging.info(f'''Starting Epoch [{epoch+1}/{num_epochs}]''')
        epoch_start = time.time()
        
        # Train model
        model.train()
        train_loss = train_epoch(train_iter, model, optim, batch_size, tgt_tokenizer, SRC, TGT, True)
        
        model.eval()
        with torch.no_grad():
            valid_loss = train_epoch(val_iter, model, optim, 1, tgt_tokenizer, SRC, TGT, False)
        
        epoch_time = time.time() - epoch_start
        
        logging.info(f'''Epoch [{epoch+1}/{num_epochs}] complete in {epoch_time:.3f} seconds.''')
        logging.info(f'''Train Loss: {train_loss:.3f}. Val Loss: {valid_loss:.3f}''')
        
        if valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            
            logging.info("Saving state dict")
            torch.save(model.state_dict(), checkpoint_file)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Check Example after each epoch:
        logging.info('')
        logging.info(f"Source: {src_tokenizer.decode(test_src_sentence, BLANK_WORD)}")
        logging.info('')
        logging.info(f"Target: {tgt_tokenizer.decode(test_tgt_sentence, BLANK_WORD)}")
        logging.info('')
        logging.info(f"Predicted: {greedy_decode_sentence(model, test_src_sentence, SRC, TGT, tgt_tokenizer)}")
        logging.info("-----------------------------------------")
        logging.info('')
    
    train_time = time.time() - train_start
    logging.info(f'''Training complete in {train_time/60/60:.3f} hours.''')
    logging.info(f'''Best train loss: {best_train_loss}. Best val loss: {best_valid_loss}. Attained at epoch {best_epoch}''')
    logging.info('')
    return train_losses, valid_losses


def main(tokenizer, src_tok_file, tgt_tok_file, train_file, val_file, test_file, num_epochs, batch_size, d_model,
         nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
         dropout, learning_rate, data_path, checkpoint_file, do_train):
    logging.info('Using tokenizer: {}'.format(tokenizer))
    
    src_tokenizer = TokenizerWrapper(tokenizer, BLANK_WORD, SEP_TOKEN, CLS_TOKEN, PAD_TOKEN, MASK_TOKEN)
    src_tokenizer.train(src_tok_file, 20000, SPECIAL_TOKENS)
    
    tgt_tokenizer = TokenizerWrapper(tokenizer, BLANK_WORD, SEP_TOKEN, CLS_TOKEN, PAD_TOKEN, MASK_TOKEN)
    tgt_tokenizer.train(tgt_tok_file, 20000, SPECIAL_TOKENS)
    
    SRC = ttdata.Field(tokenize=src_tokenizer.tokenize, pad_token=BLANK_WORD)
    TGT = ttdata.Field(tokenize=tgt_tokenizer.tokenize, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
    
    logging.info('Loading training data...')
    train_ds, val_ds, test_ds = ttdata.TabularDataset.splits(
        path=data_path, format='tsv',
        train=train_file,
        validation=val_file,
        test=test_file,
        fields=[('src', SRC), ('tgt', TGT)]
    )
    
    test_src_sentence = val_ds[0].src
    test_tgt_sentence = val_ds[0].tgt
    
    MIN_FREQ = 2
    SRC.build_vocab(train_ds.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_ds.tgt, min_freq=MIN_FREQ)
    
    logging.info(f'''SRC vocab size: {len(SRC.vocab)}''')
    logging.info(f'''TGT vocab size: {len(TGT.vocab)}''')
    
    train_iter = ttdata.BucketIterator(train_ds, batch_size=batch_size, repeat=False, sort_key=lambda x: len(x.src))
    val_iter = ttdata.BucketIterator(val_ds, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))
    test_iter = ttdata.BucketIterator(test_ds, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))
    
    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)
    
    model = Transformer(d_model=d_model,
                        nhead=nhead,
                        num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        source_vocab_length=source_vocab_length,
                        target_vocab_length=target_vocab_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    model = model.cuda()
    
    if do_train:
        train_losses,valid_losses = train(train_iter, val_iter,
                                          model, optim, num_epochs, batch_size,
                                          test_src_sentence, test_tgt_sentence,
                                          SRC, TGT, src_tokenizer, tgt_tokenizer,
                                          checkpoint_file)
    else:
        logging.info('Skipped training.')
    
    # Load best model and score test set
    logging.info('Loading best model.')
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    logging.info('Scoring the test set...')
    score_start = time.time()
    test_bleu, test_chrf = score(test_iter, model, tgt_tokenizer, SRC, TGT)
    score_time = time.time() - score_start
    logging.info(f'''Scoring complete in {score_time/60:.3f} minutes.''')
    logging.info(f'''BLEU : {test_bleu}''')
    logging.info(f'''CHRF : {test_chrf}''')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train_file', help='Training language pair file (tab-separated).')
    parser.add_argument('val_file', help='Validation language pair file (tab-separated).')
    parser.add_argument('test_file', help='Validation language pair file (tab-separated).')
    parser.add_argument('num_epochs', help='Number of epochs to train.')
    parser.add_argument('batch_size', help='Batch size.')
    parser.add_argument('tokenizer', help='The tokenizer type.')
    parser.add_argument('src_tok_file', help='The source tokenizer file.')
    parser.add_argument('tgt_tok_file', help='The target tokenizer file.')
    parser.add_argument(
        '--data_path',
        default='data',
        help='Directory containing the training files.')
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        help='Adam optimizer learning rate.')
    parser.add_argument(
        '--d_model',
        default=512,
        help='The number of expected features in the input to the transformer.')
    parser.add_argument(
        '--nhead',
        default=8,
        help='The number of attention heads.')
    parser.add_argument(
        '--num_encoder_layers',
        default=6,
        help='The number of encoder layers.')
    parser.add_argument(
        '--num_decoder_layers',
        default=6,
        help='The number of decoder layers.')
    parser.add_argument(
        '--dim_feedforward',
        default=2048,
        help='The dimension of the feedforward network model.')
    parser.add_argument(
        '--dropout',
        default=0.1,
        help='The dropout value.')
    parser.add_argument(
        '--checkpoint_file',
        default='checkpoint_best_epoch.pt',
        help='The file to save model checkpoint to.')
    parser.add_argument(
        '--log_file',
        default='train.log',
        help='The file to write logs to.')
    parser.add_argument(
        '--do_train',
        default='True',
        help='If False, skip training and only do scoring from checkpoint.')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.DEBUG,
                        filename=args.log_file,
                        format='%(message)s',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    
    main(args.tokenizer,
         args.src_tok_file,
         args.tgt_tok_file,
         args.train_file,
         args.val_file,
         args.test_file,
         int(args.num_epochs),
         int(args.batch_size),
         int(args.d_model),
         int(args.nhead),
         int(args.num_encoder_layers),
         int(args.num_decoder_layers),
         int(args.dim_feedforward),
         float(args.dropout),
         float(args.learning_rate),
         args.data_path,
         args.checkpoint_file,
         args.do_train == 'True')
    