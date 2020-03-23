from collections import namedtuple

import torch

InputExample = namedtuple("InputExample", ["words", "labels", "article_id", "offset"])
InputFeatures = namedtuple("InputFeatures", ["input_ids", "attention_mask", "labels"])


def read_examples_from_file(data_file, label_map, num=None):
    input_examples = list()
    count = 0
    with open(data_file, "r", encoding="utf-8") as file:
        data = file.readlines()
        for line in data:
            count += 1
            fields = line.split("\t")
            input_examples.append(
                InputExample(
                    words=fields[0].split(),
                    labels=[label_map[int(label)] for label in fields[1].split()],
                    article_id=fields[2],
                    offset=fields[3],
                )
            )
            if num != 0 and count == num:
                break
    return input_examples


def convert_examples_to_features(examples, tokenizer, label_map, ignore_index=None):
    # TODO: Think about max_seq_len
    # TODO: Make sure pad_token_id and the label numbers do not overlap
    input_features = list()
    ignore_index = tokenizer.pad_token_id
    for example in examples:
        tokens = [tokenizer.cls_token]
        labels = [ignore_index]

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            labels.extend([label_map[label]] + [ignore_index] * (len(word_tokens) - 1))

        tokens.append(tokenizer.sep_token)
        labels.append(ignore_index)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # TODO: Verify understanding
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(attention_mask)

        input_features.append(
            InputFeatures(
                input_ids=torch.tensor(input_ids),
                attention_mask=torch.tensor(attention_mask),
                labels=torch.tensor(labels),
            )
        )
    return input_features


