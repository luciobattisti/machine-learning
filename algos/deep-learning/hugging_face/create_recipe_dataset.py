from datasets import load_dataset, DatasetDict, Dataset

import glob
import os
import pandas as pd
import numpy as np

data_dir = "data/recipes/"

input_text = []
output_labels = []
for fname in glob.glob(os.path.join(data_dir, "*.csv")):
    with open(fname) as fp:
        data = fp.read()

        if "tue" in data or "wed" in data or "thu" in data:
            label = 0
        else:
            label = 1

        input_text.append(data)
        output_labels.append(label)

input_text = np.array(input_text)
output_labels = np.array(output_labels, dtype="int64")

print("Created input_text and output_labels")

neg_count, pos_count = np.bincount(output_labels)
print("Proportion of positive samples: {}".format(int(pos_count / (neg_count+pos_count)) * 100))

np.random.seed(0)
idx = np.arange(len(output_labels))
np.random.shuffle(idx)

output_labels = output_labels[idx]
input_text = input_text[idx]

print("Dataset is shuffled")

num_training = len(output_labels) // 2

dataset = DatasetDict({
    "train": Dataset.from_dict({
            "label":output_labels[0:num_training],
            "text":input_text[0:num_training]}
        ),
    "validation": Dataset.from_dict({
            "label":output_labels[num_training:],
            "text":input_text[num_training:]}
        )
})

print(dataset)

dataset.save_to_disk("data/recipe-classification-dataset")


