import os

TRAIN_ARTICLES_DIR = "datasets/train-articles/"
TRAIN_LABEL_DIR = "datasets/train-labels-task1-span-identification/"
TRAIN_PROCESSED_LABELS_DIR = "datasets/processed-trained-labels-SI/"

sentences = []
for filename in os.listdir(TRAIN_ARTICLES_DIR):

    with open(TRAIN_LABEL_DIR + filename.split('.')[0] + '.task1-SI.labels', 'r') as f:
        lines = f.readlines()
    indices = [x.rstrip().split('\t')[1:] for x in lines]
    indices = [[int(y) for y in x] for x in indices]
    indices.sort(key=lambda x: x[0])

    with open(TRAIN_ARTICLES_DIR + filename, 'r') as f:
        buffer = ""
        sen = ""
        start = 0
        end_counter = 0
        for i, index in enumerate(indices):
            low, high = index
            while start < low:
                c = f.read(1)
                sen += c
                if c == ' ':
                    end_counter = 0
                    buffer += "0 "
                if c == '\n':
                    if end_counter == 0:
                        buffer += "0"
                        sen = sen[:-1]
                        sentences.append('\t'.join([sen, buffer, filename, str(start - len(sen))]))
                        sen = ""
                        buffer = ""
                        end_counter += 1
                    else:
                        sen = sen[:-1]
                start += 1

            while start <= high:
                c = f.read(1)
                sen += c
                if c == ' ':
                    end_counter = 0
                    buffer += "1 "
                if c == '\n' or c == '':
                    if end_counter == 0:
                        buffer += "1"
                        sen = sen[:-1]
                        sentences.append('\t'.join([sen, buffer, filename, str(start - len(sen))]))
                        sen = ""
                        buffer = ""
                        end_counter += 1
                    else:
                        sen = sen[:-1]
                start += 1

        c = f.read(1)
        sen += c
        while c != '':
            if c == ' ':
                end_counter = 0
                buffer += "0 "
            if c == '\n':
                if end_counter == 0:
                    buffer += "0"
                    sen = sen[:-1]
                    sentences.append('\t'.join([sen, buffer, filename, str(start - len(sen))]))
                    sen = ""
                    buffer = ""
                    end_counter += 1
                else:
                    sen = sen[:-1]
            start += 1
            c = f.read(1)
            sen += c

with open("datasets/train-SI.txt", 'w') as f:
    f.writelines(["%s\n" % x for x in sentences])

print("Done.")
