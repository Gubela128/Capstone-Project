import json
from sklearn.model_selection import train_test_split

# Load the data
with open('data/data_to_split.json', 'r') as f:
    data = json.load(f)

train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

test_data, eval_data = train_test_split(temp_data, test_size=0.5, random_state=42)

with open('data/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('data/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)

with open('data/eval_data.json', 'w') as f:
    json.dump(eval_data, f, indent=4)