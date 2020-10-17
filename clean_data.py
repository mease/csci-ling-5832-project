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
        if s_toks <= max_words and t_toks <= max_words:
            X.append(s_lines[i])
            y.append(t_lines[i])
            if len(X) == max_pairs:
                break
    return X, y


def split(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10142020)
    print('Train size:', len(X_train))
    print('Test size:', len(X_test))
    return X_train, X_test, y_train, y_test


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lowercase):
    if lowercase:
        w = w.lower()
    w = unicode_to_ascii(w.strip())
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


def main(src_file, tgt_file, train_file, test_file, max_words, max_pairs, lowercase):
    X, y = reduce_data(src_file, tgt_file, max_words, max_pairs)
    X_train, X_test, y_train, y_test = split(X, y)
    write_file(train_file, X_train, y_train, lowercase)
    write_file(test_file, X_test, y_test, lowercase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src_file', help='Source language file.')
    parser.add_argument('tgt_file', help='Target language file.')
    parser.add_argument('train_file', help='Output training file to be written.')
    parser.add_argument('test_file', help='Output test file to be written.')
    parser.add_argument(
        '--max_words',
        default=30,
        help='Maximum number of words in sentence.')
    parser.add_argument(
        '--max_pairs',
        default=125_000,
        help='Maximum number of pairs to keep.')
    parser.add_argument(
        '--lowercase',
        default=False,
        help='Convert sentences to lowercase.')

    args = parser.parse_args()

    main(args.src_file,
         args.tgt_file,
         args.train_file,
         args.test_file,
         int(args.max_words),
         int(args.max_pairs),
         args.lowercase == 'True')