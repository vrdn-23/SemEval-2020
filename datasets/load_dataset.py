from collections import namedtuple

import torch
import string

InputExample = namedtuple("InputExample", ["words", "labels", "article_id", "offset"])
InputFeatures = namedtuple("InputFeatures", ["input_ids", "attention_mask", "labels"])
TestInputExample = namedtuple("TestInputExample", ["words", "article_id", "offset"])
TestInputFeatures = namedtuple("TestInputFeatures", ["input_ids", "attention_mask"])


def read_examples_from_file(data_file, label_map, num=None):
    input_examples = list()
    count = 0
    with open(data_file, "r", encoding="utf-8") as file:
        data = file.readlines()
        for line in data:
            count += 1
            fields = line.split("\t")
            if len(fields) == 4:
                input_examples.append(
                    InputExample(
                        words=fields[0].split(),
                        labels=[label_map[int(label)] for label in fields[1].split()],
                        article_id=fields[2],
                        offset=fields[3],
                    )
                )
            elif len(fields) == 3:
                input_examples.append(
                    TestInputExample(
                        words=fields[0].split(),
                        article_id=fields[1],
                        offset=fields[2],
                    )
                )
            if num != 0 and count == num:
                break
    return input_examples


def convert_examples_to_features(examples, tokenizer, label_map, ignore_index, max_seq_len):
    input_features = list()
    # ignore_index = tokenizer.pad_token_id

    for example in examples:
        tokens = [tokenizer.cls_token]
        labels = [ignore_index]

        for i in (zip(example.words, example.labels) if isinstance(example, InputExample) else example.words):
            if isinstance(example, InputExample):
                word, label = i
            else:
                word = i
            if not word or (len(word) == 1 and ord(word) in [65279, 8207]):
                continue
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if isinstance(example, InputExample):
                labels.extend([label_map[label]] + [ignore_index] * (len(word_tokens) - 1))

        tokens.append(tokenizer.sep_token)
        if isinstance(example, InputExample):
            labels.append(ignore_index)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # TODO: Verify understanding
        attention_mask = [1] * len(input_ids)

        if isinstance(example, InputExample):
            assert len(input_ids) == len(labels)
        assert len(input_ids) == len(attention_mask)

        padding = max_seq_len - len(input_ids)
        if padding < 0:
            input_ids = input_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]
            if isinstance(example, InputExample):
                labels = labels[:max_seq_len]
        else:
            input_ids = input_ids + [0]*padding
            attention_mask = attention_mask + [0]*padding
            if isinstance(example, InputExample):
                labels = labels + [ignore_index]*padding

        if isinstance(example, InputExample):
            input_features.append(
                InputFeatures(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(attention_mask),
                    labels=torch.tensor(labels),
                )
            )
        elif isinstance(example, TestInputExample):
            input_features.append(
                TestInputFeatures(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(attention_mask),
                )
            )
    return input_features


def custom_decode(tokenizer, input_ids, labels, test_examples):
    final_words = sum([x.words for x in test_examples], [])
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    assert len(tokens) == len(labels)
    filtered_labels = []
    filtered_tokens = []
    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue
        filtered_tokens.append(token)
        filtered_labels.append(label.item())
    final_labels = []
    final_tokens = []
    for token, label in zip(filtered_tokens, filtered_labels):
        if "#" in token or token in string.punctuation or (
                len(token) == 1 and ord(token) in [8217, 8220, 8221, 8211, 8212, 115, 116, 8216, 8230]):
            continue
        final_labels.append(label)
        final_tokens.append(token)
    final_token_count = 0
    final_final_labels = []
    final_final_words = []
    check_next = False
    i = 0
    total_count = 0
    while i < len(final_words):
        total_count += 1
        if final_tokens[final_token_count] in ["https", "http"]:
            final_final_labels.append(final_labels[final_token_count])
            while final_tokens[final_token_count] in final_words[i].lower():
                final_token_count += 1
            i += 1
            pass
        #print(i, " ==> ", tuple([final_tokens[final_token_count], final_words[i]]), " ==> ", len(final_final_labels))
        if final_tokens[final_token_count] in final_words[i].lower():
            final_final_words.append(final_words[i])
            final_final_labels.append(final_labels[final_token_count])
            if check_next:
                final_final_labels.append(final_labels[final_token_count])
                check_next = False
            final_token_count += 1
        else:
            # takes care of A.D. case
            if final_tokens[final_token_count] in final_words[i - 1].lower() and final_tokens[final_token_count] != \
                    final_tokens[final_token_count - 1]:
                final_token_count += 1
                continue
            # take care of alas case
            elif i + 1 < len(final_words) and final_tokens[final_token_count] in final_words[i + 1].lower():
                check_next = True
            elif final_words[i] in string.punctuation + "___!’" or (
                    len(final_words[i]) == 1 and ord(final_words[i]) in [8217, 8220, 8221, 8211, 8212, 8216, 8230]):
                final_final_labels.append(final_final_labels[-1])
        i += 1

    assert len(final_words) == len(final_final_labels)

    return final_final_labels


def write_labels_to_file(filename, labels, test_examples):
    label_count = 0
    with open(filename, 'w') as f:
        for example in test_examples:
            label_string = str(labels[label_count])
            label_count += 1
            for i in example.words[1:]:
                label_string += " " + str(labels[label_count])
                label_count += 1
            f.write(" ".join(example.words) + "\t" + label_string + "\t" + example.article_id + "\t" + example.offset)
    assert label_count == len(sum([x.words for x in test_examples], []))