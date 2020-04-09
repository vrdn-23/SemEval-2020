from argparse import ArgumentParser
import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from load_dataset import read_examples_from_file, convert_examples_to_features, InputFeatures, \
    custom_decode, write_labels_to_file
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
import time
import numpy as np
from tqdm import tqdm

ignore_index = -100


def custom_collate_fn(batch):
    input_ids_batch, attention_mask_batch, labels_batch = list(), list(), list()
    for sample in batch:
        input_ids, attention_mask = sample.input_ids, sample.attention_mask
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        if isinstance(sample, InputFeatures):
            labels = sample.labels
            labels_batch.append(labels)

    if isinstance(batch[0], InputFeatures):
        return torch.stack(input_ids_batch), torch.stack(attention_mask_batch), torch.stack(labels_batch)
    else:
        return torch.stack(input_ids_batch), torch.stack(attention_mask_batch)


# def inference(model, test_loader, device, tokenizer, test_examples):
#     prediction_labels = []
#
#     with torch.no_grad():
#         model.eval()
#         for iteration, (input_ids, attention_mask) in tqdm(enumerate(test_loader)):
#             input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
#             scores, *_ = model(input_ids=input_ids, attention_mask=attention_mask)
#             flattened_input_ids = input_ids.flatten()
#             prediction = torch.max(scores, 2)[1].flatten()
#             print(torch.unique(prediction))
#             prediction_labels.extend(custom_decode(tokenizer, flattened_input_ids, prediction, test_examples[iteration * test_loader.batch_size: (iteration + 1) * test_loader.batch_size]))
#
#     return prediction_labels


def train(model, train_loader, dev_loader, args, optimizer, device, scheduler, test_loader, tokenizer, test_examples, SI_labels_inv):
    best_dev_loss = np.inf
    best_epoch = 0
    optimizer.zero_grad()

    for epoch in range(args.num_train_epochs):
        start = time.time()
        model.train()
        train_loss = 0

        for iteration, (input_ids, attention_mask, labels) in tqdm(enumerate(train_loader)):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            loss, *_ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()

            if (iteration + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        with torch.no_grad():
            model.eval()
            dev_loss = 0

            label_ids = list()
            preds_list = list()

            for iteration, (input_ids, attention_mask, labels) in tqdm(enumerate(dev_loader)):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                loss, scores = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                dev_loss += loss.item()

                predictions = torch.max(scores, 2)[1]
                batch_size, max_seq_len = input_ids.shape
                for b in range(batch_size):
                    label_ids_temp = []
                    preds_list_temp = []
                    for l in range(max_seq_len):
                        if labels[b, l] != -100:
                            label_ids_temp.append(SI_labels_inv[labels[b, l].item()])
                            preds_list_temp.append(SI_labels_inv[predictions[b, l].item()])
                    label_ids.append(label_ids_temp.copy())
                    preds_list.append(preds_list_temp.copy())

            dev_loss /= len(dev_loader)
            print('Train Loss: {:.4f}\tVal Loss: {:.4f}'.format(train_loss, dev_loss))
            print("Precision: " + str(precision_score(label_ids, preds_list)))
            print("Recall: " + str(recall_score(label_ids, preds_list)))
            print("F1: " + str(f1_score(label_ids, preds_list)))

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch

        # predicted_labels = inference(model, test_loader, device, tokenizer, test_examples)
        # write_labels_to_file(args.sub_file + str(epoch) + ".txt", predicted_labels, test_examples)

        torch.save(model.state_dict(), args.model_path + 'model' + str(epoch) + '.pt')
        torch.save(optimizer.state_dict(), args.model_path + 'optimizer' + str(epoch) + '.pt')
        print('Time taken ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch))

    print('Least validation error {:.4f} at epoch {}'.format(best_dev_loss, best_epoch))


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default="train-SI.txt", help="Path of the training data.")
    parser.add_argument("--dev_file", type=str, default="", help="Path of validation data.")
    parser.add_argument("--test_file", type=str, default="", help="Path of test file (without labels)")
    parser.add_argument("--sub_file", type=str, default="", help="Path of test file (with labels)")
    parser.add_argument("--model_path", type=str, default="", help="Path to store models")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--random_n", type=int, default=0, help="Small dataset size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    args = parser.parse_args()

    max_seq_len = 200

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # TODO: Extend entire code from here to TC

    SI_labels = dict()
    SI_labels["0"] = 0
    SI_labels["1"] = 1
    SI_labels_inv = {v: k for k,v in SI_labels.items()}

    input_examples = read_examples_from_file(args.data_file, SI_labels, args.random_n)
    train_data = convert_examples_to_features(input_examples, tokenizer, SI_labels, ignore_index, max_seq_len)

    valid_examples = read_examples_from_file(args.dev_file, SI_labels, args.random_n)
    dev_data = convert_examples_to_features(valid_examples, tokenizer, SI_labels, ignore_index, max_seq_len)

    # test_examples = read_examples_from_file(args.test_file, SI_labels, args.random_n)
    # test_data = convert_examples_to_features(test_examples, tokenizer, SI_labels, ignore_index, max_seq_len)

    train_sampler = RandomSampler(train_data)
    dev_sampler = RandomSampler(dev_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, sampler=dev_sampler, collate_fn=custom_collate_fn)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print("Data Loaders ready")

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # training
    # TODO: Send test at correct places when actually testing
    train(model, train_loader, dev_loader, args, optimizer, device, scheduler, dev_loader, tokenizer, valid_examples, SI_labels_inv)

    return 0


if __name__ == '__main__':
    main()
