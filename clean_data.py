import argparse
import io
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import unicodedata
import re

def reduce_data(src_file, tgt_file, max_words, max_pairs):
    # Get first MAX_PAIRS pairs that are less than MAX_WORDS words.
    s_lines = io.open(src_file, encoding='UTF-8').read().strip().split('\n')
    t_lines = io.open(tgt_file, encoding='UTF-8').read().strip().split('\n')
    
    X = []
    y = []
    
    for i in range(len(s_lines)):
        s_toks = len(word_tokenize(s_lines[i]))
        t_toks = len(word_tokenize(t_lines[i]))
        if s_toks <= max_words and t_toks <= max_words and s_toks !=0 and t_toks != 0:
            X.append(s_lines[i])
            y.append(t_lines[i])
            if len(X) == max_pairs:
                break
    return X, y


def split(X, y, train, val, test):
    ratio = (val + test) / (train + val + test)
    X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=ratio, random_state=10142020)
    
    ratio = test / (val + test)
    X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=ratio, random_state=10152020)
    print('Train size:', len(X_train))
    print('Val size:  ', len(X_val))
    print('Test size: ', len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lowercase):
    w = unicode_to_ascii(w.strip())
    if lowercase:
        w = w.lower()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"([\t]+)", " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return w


def write_file(out_file, X, y, lowercase):
    with io.open(out_file, 'w', encoding='UTF-8') as f:
        for i in range(len(X)):
            f.write("{}\t{}\n".format(preprocess_sentence(X[i], lowercase),
                                      preprocess_sentence(y[i], lowercase)))
    print("Wrote", out_file)

def write_tok_file(tok_file, data, lowercase):
    with io.open(tok_file, 'w', encoding='UTF-8') as f:
        for i in range(len(data)):
            f.write('{}\n'.format(preprocess_sentence(data[i], lowercase)))
    print("Wrote", tok_file)
    

def main(src_file, tgt_file, train_file, val_file, test_file, src_tok_file,
         tgt_tok_file, max_words, train, val, test, lowercase):
    max_pairs = train + val + test
    X, y = reduce_data(src_file, tgt_file, max_words, max_pairs)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y, train, val, test)
    write_file(train_file, X_train, y_train, lowercase)
    write_file(val_file, X_val, y_val, lowercase)
    write_file(test_file, X_test, y_test, lowercase)
    write_tok_file(src_tok_file, X_train, lowercase)
    write_tok_file(tgt_tok_file, y_train, lowercase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src_file', help='Source language file.')
    parser.add_argument('tgt_file', help='Target language file.')
    parser.add_argument('train_file', help='Output training file to be written.')
    parser.add_argument('val_file', help='Output validation file to be written.')
    parser.add_argument('test_file', help='Output test file to be written.')
    parser.add_argument('src_tok_file', help='Source language file for tokenizer training to be written.')
    parser.add_argument('tgt_tok_file', help='Target language file for tokenizer training to be written.')
    parser.add_argument(
        '--max_words',
        default=30,
        help='Maximum number of words in sentence.')
    parser.add_argument(
        '--train',
        default=100_000,
        help='Number of training samples.')
    parser.add_argument(
        '--val',
        default=25_000,
        help='Number of validation samples.')
    parser.add_argument(
        '--test',
        default=25_000,
        help='Number of test samples.')
    parser.add_argument(
        '--lowercase',
        default='True',
        help='Convert sentences to lowercase.')

    args = parser.parse_args()

    main(args.src_file,
         args.tgt_file,
         args.train_file,
         args.val_file,
         args.test_file,
         args.src_tok_file,
         args.tgt_tok_file,
         int(args.max_words),
         int(args.train),
         int(args.val),
         int(args.test),
         args.lowercase == 'True')