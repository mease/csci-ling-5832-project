import nltk
from tokenizers import ByteLevelBPETokenizer

class TokenizerWrapper:
    def __init__(self, tok_type):
        self.tok_type = tok_type
        
        if self.tok_type == 'bpe':
            self.tokenizer = ByteLevelBPETokenizer()
    
    def train(self, data_file, vocab_size, special_tokens):
        if self.tok_type == 'bpe':
            self.tokenizer.train([data_file], vocab_size=vocab_size, special_tokens=special_tokens)

    def tokenize(self, text):
        if self.tok_type == 'bpe':
            return self.tokenizer.encode(text).tokens
        elif self.tok_type == 'word':
            return nltk.tokenize.word_tokenize(text)
        else:
            raise Exception('Unknown tokenizer: ' + self.tok_type)
    
    def decode(self, text, blank_token):
        if self.tok_type == 'bpe':
            tokens = text.strip().split(' ')
            ids = [self.tokenizer.token_to_id(t) for t in tokens]
            ids = [i if i != None else self.tokenizer.token_to_id(blank_token) for i in ids]
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        elif self.tok_type == 'word':
            return text.strip()
        else:
            raise Exception('Unknown tokenizer: ' + self.tok_type)