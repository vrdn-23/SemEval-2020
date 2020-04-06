from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForTokenClassification
from tools.utils.load_dataset import read_examples_from_file, convert_examples_to_features, InputFeatures, custom_decode, write_labels_to_file
#from load_dataset import read_examples_from_file, convert_examples_to_features, InputFeatures, \
    #custom_decode, \
    #write_labels_to_file
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from tqdm import tqdm

ignore_index = -100
PAD = 0


class customDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def custom_collate_fn(batch):
    input_ids_batch, attention_mask_batch, labels_batch = list(), list(), list()
    for sample in batch:
        input_ids, attention_mask = sample.input_ids, sample.attention_mask
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        if isinstance(sample, InputFeatures):
            labels = sample.labels
            labels_batch.append(labels)

    input_ids = pad_sequence(input_ids_batch, batch_first=True, padding_value=PAD)
    attention_mask = pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
    assert len(input_ids) == len(attention_mask)
    if isinstance(batch[0], InputFeatures):
        labels = pad_sequence(labels_batch, batch_first=True, padding_value=ignore_index)
        assert len(input_ids) == len(labels)
        return input_ids, attention_mask, labels
    else:
        return input_ids, attention_mask


def inference(model, test_loader, device, tokenizer, test_examples):
    prediction_labels = []
    with torch.no_grad():
        model.eval()
        for iteration, (input_ids, attention_mask) in tqdm(enumerate(test_loader)):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            scores, *_ = model(input_ids=input_ids, attention_mask=attention_mask)
            flattened_input_ids = input_ids.flatten()
            prediction = torch.max(scores, 2)[1].flatten()
            prediction_labels.extend(custom_decode(tokenizer, flattened_input_ids, prediction,
                                                   test_examples[iteration * test_loader.batch_size: (
                                                                                                             iteration + 1) * test_loader.batch_size]))

    return prediction_labels


def train(model, train_loader, dev_loader, args, optimizer, device, scheduler, test_loader, tokenizer, test_examples):
    best_dev_loss = np.inf
    best_epoch = 0
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss = 0

        for iteration, (input_ids, attention_mask, labels) in tqdm(enumerate(train_loader)):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # scheduler.step(epoch + iteration / len(train_loader))
            scores, *_ = model(input_ids=input_ids, attention_mask=attention_mask)

            scores_flattened = scores.view(-1, scores.shape[-1])
            labels_flattened = labels.view(-1)
            prediction = torch.max(scores, 2)[1].flatten()
            print(torch.unique(prediction))

            # TODO: Check reduction across batch and across sentence and also, overall formula
            loss = criterion(scores_flattened, labels_flattened)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        with torch.no_grad():
            model.eval()
            dev_loss = 0
            for iteration, (input_ids, attention_mask, labels) in tqdm(enumerate(dev_loader)):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                scores, *_ = model(input_ids=input_ids, attention_mask=attention_mask)

                scores_flattened = scores.view(-1, scores.shape[-1])
                labels_flattened = labels.view(-1)

                loss = criterion(scores_flattened, labels_flattened)
                dev_loss += loss.item()

            dev_loss /= len(dev_loader)
            print('Train Loss: {:.4f}\tVal Loss: {:.4f}'.format(train_loss, dev_loss))

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            predicted_labels = inference(model, test_loader, device, tokenizer, test_examples)
            write_labels_to_file(args.sub_file + str(epoch) + ".txt", predicted_labels, test_examples)

            torch.save(model.state_dict(), args.model_path + 'model' + str(epoch) + '.pt')
            torch.save(optimizer.state_dict(), args.model_path + 'optimizer' + str(epoch) + '.pt')
        print('Time taken ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch))

    print('Least validation error {:.4f} at epoch {}'.format(best_dev_loss, best_epoch))


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default="", help="Path of the training data.")
    parser.add_argument("--dev_file", type=str, default="", help="Path of validation data.")
    parser.add_argument("--test_file", type=str, default="", help="Path of test file (without labels)")
    parser.add_argument("--sub_file", type=str, default="", help="Path of test file (with labels)")
    parser.add_argument("--model_path", type=str, default="", help="Path to store models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--random_n", type=int, default=0, help="Small dataset size")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained("bert-base-uncased")
    model.to(device)
    tokenizer

    # TODO: Extend entire code from here to TC

    # TODO: Avoid this hard coding
    SI_labels = dict()
    SI_labels[0] = 0
    SI_labels[1] = 1
    SI_inv_labels = {value: key for key, value in SI_labels.items()}

    input_examples = read_examples_from_file(args.data_file, SI_labels, args.random_n)
    train_data = convert_examples_to_features(input_examples, tokenizer, SI_labels, ignore_index)

    valid_examples = read_examples_from_file(args.dev_file, SI_labels, args.random_n)
    dev_data = convert_examples_to_features(valid_examples, tokenizer, SI_labels, ignore_index)

    test_examples = read_examples_from_file(args.test_file, SI_labels, args.random_n)
    test_data = convert_examples_to_features(test_examples, tokenizer, SI_labels, ignore_index)

    train_dataset, dev_dataset, test_dataset = customDataset(train_data), customDataset(dev_data), customDataset(
        test_data)
    # test_dataset = customDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print("Data Loaders ready")

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)

    train(model, train_loader, dev_loader, args, optimizer, device, scheduler, test_loader, tokenizer, test_examples)

    return 0


if __name__ == '__main__':
    main()
