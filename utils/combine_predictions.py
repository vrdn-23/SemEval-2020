dev_file = "datasets/dev-SI.txt"
dev_predicted = "datasets/test_predictions.txt"
dev_submission = "datasets/dev_submission-SI.txt"

dev = open(dev_file, 'r')
dev_sentences = dev.readlines()
dev_p = open(dev_predicted, 'r')

dev_p_sentences = []
labels = []
org_dev_words = []
total_org_sentences = 0
for line in dev_sentences:
    words, _, _ = line.split("\t")
    org_dev_words += words.split() + ["**##**"]
    total_org_sentences += 1

org_dev_count = 0
pred_dev_count = 0
pred_dev_words = dev_p.readlines()

final_labels = []
total_pred_sentences = 0
prev_new_line = False
for line in pred_dev_words:
    if line == "\n":
        if prev_new_line:
            continue
        final_labels.append("**##**")
        total_pred_sentences += 1
        prev_new_line = True
    else:
        temp = tuple(line.split())
        final_labels.append(temp)
        prev_new_line = False

print(total_org_sentences)
print(total_pred_sentences)
print(len(final_labels))
#print(org_dev_words)

print(len(org_dev_words), len(pred_dev_words))
assert len(org_dev_words) == len(pred_dev_words)

for line1, line2 in zip(dev_sentences, pred_dev_words):
    if line1 == "\n":
        # assert len(words) == len(labels)
        dev_p_sentences.append(labels)
        # words = []
        labels = []
    else:
        word, label = line1.rstrip().split()
        # words += list(word)
        labels += list(label)

# if labels:
#     dev_p_sentences.append(labels)
print(dev_p_sentences[-1])
dev_p_sentences = dev_p_sentences[:-1]

print(len(dev_p_sentences))
print(len(dev_sentences))

count = 1
for line1, line2 in zip(dev_sentences, dev_p_sentences):
    print(count)
    a, b, c = line1.rstrip().split('\t')
    words = a.split()
    if len(words) != len(line2):
        print(len(words), len(line2))
        print(words)
        print(line2)
    assert len(words) == len(line2)

    count += 1

assert len(dev_p_sentences) == len(dev_sentences)

new_lines = []
for line1, line2 in zip(dev_sentences, dev_p_sentences):
    a, b, c = line1.rstrip().split('\t')
    d = " ".join(line2)
    s = "\t".join([a, d, b, c]) + "\n"
    new_lines.append(s)

dev_s = open(dev_submission, 'w')
dev_s.writelines(new_lines)
