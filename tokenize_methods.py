import nltk
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer

class TokenizerWrapper:
    def __init__(self, tok_type, unk_token, sep_token, cls_token, pad_token, mask_token):
        self.tok_type = tok_type
        
        if self.tok_type == 'bpe':
            self.tokenizer = ByteLevelBPETokenizer()
        elif self.tok_type == 'wordpiece':
            self.tokenizer = BertWordPieceTokenizer(unk_token=unk_token,
                                                    sep_token=sep_token,
                                                    cls_token=cls_token,
                                                    pad_token=pad_token,
                                                    mask_token=mask_token)
        elif self.tok_type == 'sentencepiece':
            self.tokenizer = SentencePieceBPETokenizer(unk_token=unk_token)
    
    def train(self, data_file, vocab_size, special_tokens):
        if self.tok_type in ['bpe', 'wordpiece', 'sentencepiece']:
            self.tokenizer.train([data_file], vocab_size=vocab_size, special_tokens=special_tokens)

    def tokenize(self, text):
        if self.tok_type in ['bpe', 'wordpiece', 'sentencepiece']:
            return self.tokenizer.encode(text).tokens
        elif self.tok_type == 'word':
            return nltk.tokenize.word_tokenize(text)
        elif self.tok_type == 'char':
            return [c for c in text]
        else:
            raise Exception('Unknown tokenizer: ' + self.tok_type)
    
    def decode(self, tokens, blank_token):
        if self.tok_type in ['bpe', 'wordpiece', 'sentencepiece']:
            ids = [self.tokenizer.token_to_id(t) for t in tokens]
            ids = [i if i != None else self.tokenizer.token_to_id(blank_token) for i in ids]
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        elif self.tok_type == 'word':
            return ' '.join(tokens)
        elif self.tok_type == 'char':
            return ''.join(tokens)
        else:
            raise Exception('Unknown tokenizer: ' + self.tok_type)