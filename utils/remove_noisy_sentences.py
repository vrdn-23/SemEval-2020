dev_file = "datasets/dev-TC.txt"
train_file = "datasets/train-TC.txt"
new_dev_file = "datasets/dev-TC-2.txt"
new_train_file = "datasets/train-TC-2.txt"


dev = open(dev_file, 'r')
train = open(train_file, 'r')

new_d = []
new_t = []
d_count = 0
total_d_count = 0
t_count = 0
total_t_count = 0
for line in dev.readlines():
    sentence, labels, _, _ = line.rstrip().split('\t')
    length = len(sentence.split())
    if length < 3:
        if "1" in labels:
            d_count += 1
        total_d_count += 1
        continue
    new_d += line
for line in train.readlines():
    sentence, labels, _, _ = line.rstrip().split('\t')
    length = len(sentence.split())
    if length < 3:
        if "1" in labels:
            t_count += 1
        total_t_count += 1
        continue
    new_t += line

new_dev = open(new_dev_file, 'w')
new_dev.writelines(new_d)
new_train = open(new_train_file, 'w')
new_train.writelines(new_t)

print(d_count, total_d_count, t_count, total_t_count)

dev_file = "datasets/dev-SI.txt"
train_file = "datasets/train-SI.txt"
new_dev_file = "datasets/dev-SI-2.txt"
new_train_file = "datasets/train-SI-2.txt"


dev = open(dev_file, 'r')
train = open(train_file, 'r')

new_d = []
new_t = []
d_count = 0
total_d_count = 0
t_count = 0
total_t_count = 0
for line in dev.readlines():
    sentence, labels, _, _ = line.rstrip().split('\t')
    length = len(sentence.split())
    if length < 3:
        if "1" in labels:
            d_count += 1
        total_d_count += 1
        continue
    new_d += line
for line in train.readlines():
    sentence, labels, _, _ = line.rstrip().split('\t')
    length = len(sentence.split())
    if length < 3:
        if "1" in labels:
            t_count += 1
        total_t_count += 1
        continue
    new_t += line

new_dev = open(new_dev_file, 'w')
new_dev.writelines(new_d)
new_train = open(new_train_file, 'w')
new_train.writelines(new_t)

print(d_count, total_d_count, t_count, total_t_count)

