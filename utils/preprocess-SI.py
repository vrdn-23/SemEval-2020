import os

TRAIN_ARTICLES_DIR = "datasets/train-articles/"
TRAIN_LABEL_DIR = "datasets/train-labels-task1-span-identification/"
TRAIN_PROCESSED_LABELS_DIR = "datasets/processed-trained-labels-SI/"

punc_list = [' ', ',', '"', ";", ":", "!", "?", "'"]

for filename in os.listdir(TRAIN_ARTICLES_DIR):
    if filename != "article0000001.txt":
        continue
    with open(TRAIN_LABEL_DIR + filename.split('.')[0] + '.task1-SI.labels', 'r') as f:
        lines = f.readlines()
    indices = [x.rstrip().split('\t')[1:] for x in lines]
    indices = [[int(y) for y in x] for x in indices]
    indices.sort(key=lambda x: x[0])

    with open(TRAIN_ARTICLES_DIR + filename, 'r') as f:
        buffer = ""
        start = 0
        end_counter = 0
        for i, index in enumerate(indices):
            low, high = index
            while start < low:
                c = f.read(1)
                if c in punc_list:
                    end_counter = 0
                    if c == ' ':
                        buffer += "0 "
                if c == '\n':
                    if end_counter == 0:
                        buffer += "0"
                        end_counter += 1
                    buffer += '\n'
                start += 1

            while start <= high:
                c = f.read(1)
                if c in punc_list:
                    end_counter = 0
                    if c == ' ' or start == high:
                        buffer += "1 "
                if c == '\n' or c == '':
                    if end_counter == 0:
                        buffer += "1"
                        end_counter += 1
                    buffer += '\n'
                start += 1

        c = f.read(1)
        while c != '':
            if c in punc_list:
                end_counter = 0
                if c == ' ':
                    buffer += "0 "
            if c == '\n':
                if end_counter == 0:
                    buffer += "0"
                    end_counter += 1
                buffer += '\n'
            start += 1
            c = f.read(1)

    with open(TRAIN_PROCESSED_LABELS_DIR + filename, 'w') as f:
        f.write(buffer)

print("Done.")
