import os

TRAIN_ARTICLES_DIR = "datasets/dev-articles/"
TRAIN_LABEL_DIR = "datasets/gold-dev/dev-labels-task2-technique-classification/"
TRAIN_PROCESSED_LABELS_DIR = "datasets/processed-trained-labels-TC/"

res = open("datasets/dev-TC.txt", 'w')

for filename in os.listdir(TRAIN_ARTICLES_DIR):

    with open(TRAIN_LABEL_DIR + filename.split('.')[0] + '.task2-TC.labels', 'r') as f:
        lines = f.readlines()

    indices_techniques = [x.rstrip().split('\t')[1:] for x in lines]
    indices = [[x[0], int(x[1]), int(x[2])] for x in indices_techniques]
    indices.sort(key=lambda x: x[1])

    with open(TRAIN_ARTICLES_DIR + filename, 'r') as f:
        offset = 0
        index = 0

        if len(indices) == 0:
            continue

        for line in f:
            sentence_offset = offset
            labels = []
            words = line.rstrip().split(" ")
            if len(words) == 1 and words[0] == '':
                offset += 1
                continue

            for word in words:
                if index < len(indices):
                    low = indices[index][1]
                    high = indices[index][2]
                else:
                    index = len(indices) - 1

                word_starting_off = offset
                word_ending_off = offset + len(word) - 1

                if word_ending_off < low:
                    # Mark word as 0 :  selh
                    labels.append("0")
                elif word_starting_off <= low <= word_ending_off:
                    # sleh, slhe
                    labels.append(indices[index][0])
                elif word_starting_off >= low:
                    if word_starting_off < high:
                        # lseh
                        labels.append(indices[index][0])
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
