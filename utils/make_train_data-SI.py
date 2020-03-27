import os

# TRAIN_ARTICLES_DIR = "datasets/train-articles/"
# TRAIN_LABEL_DIR = "datasets/train-labels-task1-span-identification/"
# res = open("datasets/train-SI.txt", 'w')

TRAIN_ARTICLES_DIR = "datasets/dev-articles/"
TRAIN_LABEL_DIR = "datasets/gold-dev/dev-labels-task1-span-identification/"
res = open("datasets/dev-SI.txt", 'w')


for filename in os.listdir(TRAIN_ARTICLES_DIR):

    with open(TRAIN_LABEL_DIR + filename.split('.')[0] + '.task1-SI.labels', 'r') as f:
        lines = f.readlines()

    indices = [x.rstrip().split('\t')[1:] for x in lines]
    indices = [[int(y) for y in x] for x in indices]
    indices.sort(key=lambda x: x[0])

    with open(TRAIN_ARTICLES_DIR + filename, 'r') as f:
        offset = 0
        index = 0

        for line in f:
            line = line.replace(chr(8212), " ")
            line = line.replace("-", " ")
            sentence_offset = offset
            labels = []
            words = line.rstrip().split(" ")
            if len(words) == 1 and words[0] == '':
                offset += 1
                continue

            for word in words:
                if not word:
                    offset += 1
                    continue
                if index < len(indices):
                    low = indices[index][0]
                    high = indices[index][1]

                word_starting_off = offset
                word_ending_off = offset + len(word) - 1

                if word_ending_off < low:
                    # Mark word as 0 :  selh
                    labels.append("0")
                elif word_starting_off <= low <= word_ending_off:
                    # sleh, slhe
                    labels.append("1")
                elif word_starting_off >= low:
                    if word_starting_off < high:
                        # lseh
                        labels.append("1")
                    else:
                        # lhse
                        labels.append("0")
                else:
                    assert False

                offset = word_ending_off + 2
                if high <= offset:
                    index += 1

            res.write("\t".join([" ".join(words), " ".join(labels), filename, str(sentence_offset)]))
            res.write("\n")

res.close()
print("Done.")
