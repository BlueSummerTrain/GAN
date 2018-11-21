import numpy as np
import codecs
import re
import os
from collections import defaultdict
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def build_vocabulary(sentences, savedir):
    word_dict = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            if word != '<unk>':
                word_dict[word] = +1
            else:
                word_dict[word] = 0
    voc = dict()
    word_list = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    for i, iterm in enumerate(word_list):
        index = i + 1
        key = iterm[0]
        voc[key] = index
    with open(savedir, 'wb') as file:
        pickle.dump(voc, file)
    return voc, len(voc.keys())


def read_vocabulary(voc_dir):
    voc_file = open(voc_dir, 'r')
    voc = pickle.load(voc_file)
    print 'read vocabulary len : %f' % len(voc.keys())
    return voc, len(voc.keys())


def sentence2matrix(sentences, max_length, vocs):
    sentences_num = len(sentences)
    data_dict = np.zeros((sentences_num, max_length), dtype='int32')
    for index, sentence in enumerate(sentences):
        data_dict[index, :] = map2id(sentence, vocs, max_length)
    return data_dict


def map2id(sentence, voc, max_len):
    array_int = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(sentence))
    for i in range(min_range):
        iterm = sentence[i]
        array_int[i] = voc.get(iterm, voc['<unk>'])
    return array_int


def clean_str(string):
    string = re.sub(ur"[^\u4e00-\u9fffA-Za-z0-9:,.!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def read_and_clean_file(input_file, output_cleaned_file=None):
    lines = list(open(input_file, "r").readlines())
    lines = [clean_str(seperate_line(line.decode('utf-8'))) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines


def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = read_and_clean_file(positive_data_file)
    negative_examples = read_and_clean_file(negative_data_file)
    x_text = positive_examples + negative_examples
    negative_labels = [[1, 0] for _ in negative_examples]
    positive_labels = [[0, 1] for _ in positive_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, positive_examples, y]


def load_testfile_and_labels(input_text_file, input_label_file, num_labels):
    x_text = read_and_clean_file(input_text_file)
    y = None if not os.path.exists(input_label_file) else map(
        int, list(open(input_label_file, "r").readlines()))
    return (x_text, y)


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] *
                            (max_sentence_length - len(sentence)))
    return sentences


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def read_data_from_strs(lines, max_sentence_length):
    data_line = []
    for line in lines:
        line = line.decode('utf-8')
        line = ''.join([word + ' ' for word in line])
        line = re.sub(ur"[^\u4e00-\u9fffA-Za-z0-9:,.!?\'\`]", " ", line)
        line = re.sub(r"\s{2,}", " ", line)
        line = line.strip().lower()
        line = line.split(' ')
        if len(line) > max_sentence_length:
            line = line[:max_sentence_length]
        else:
            line.extend(['<unk>'] * (max_sentence_length - len(line)))
        data_line.append(line)
    return data_line


def read_data_from_str(line, max_sentence_length):
    line = line.decode('utf-8')
    line = ''.join([word + ' ' for word in line])
    line = re.sub(ur"[^\u4e00-\u9fffA-Za-z0-9:,.!?\'\`]", " ", line)
    line = re.sub(r"\s{2,}", " ", line)
    line = line.strip().lower()
    line = line.split(' ')
    if len(line) > max_sentence_length:
        line = line[:max_sentence_length]
    else:
        line.extend(['<unk>'] * (max_sentence_length - len(line)))
    return [line]


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict
