import json
from sklearn.model_selection import train_test_split

with open('/home/toefl/K/nto/dataset/labels.json') as f:
    train_data = json.load(f)

train_data = [(k, v) for k, v in train_data.items()]
print('Total len:', len(train_data))

# Drop a couple of too long samples
train_data = [i for i in train_data if len(i[1]) < 19]

counts = [0 for i in range(25)]
for i in train_data:
    counts[len(i[1]) - 1] += 1
print(counts)

# Stratified split is based on samples lengths. Train size: 0.8. Val size: 0.2.
train_data_splitted, val_data_splitted = train_test_split(train_data, test_size=0.2, train_size=0.8,
                                                          random_state=0xFACED, shuffle=True,
                                                          stratify=[len(i[1]) for i in train_data]
                                                          )

print('Train len after split:', len(train_data_splitted))
print('Val len after split:', len(val_data_splitted))

with open('/home/toefl/K/baseline/dataset/train_labels.json', 'w') as f:
    json.dump(dict(train_data_splitted), f)

with open('/home/toefl/K/baseline/dataset/val_labels.json', 'w') as f:
    json.dump(dict(val_data_splitted), f)