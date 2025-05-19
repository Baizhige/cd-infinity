import numpy as np
import os
file_name_X = "balanced_data.npy"
file_name_Y = "balanced_label.npy"

base_path = "to/your/dataset"
input_path_X = os.path.join(base_path, "all", file_name_X)
input_path_Y = os.path.join(base_path, "all", file_name_Y)

for sub_dir in ["train", "eval", "test"]:
    os.makedirs(os.path.join(base_path, sub_dir), exist_ok=True)

all_data = np.load(input_path_X)

all_label = np.load(input_path_Y)
len_data = all_data.shape[0]

# === shuffle
indices = np.random.permutation(len_data)
all_data = all_data[indices]
all_label = all_label[indices]
# === shuffle

split_len = int(len_data / 10)

test_list = [
    (0 * split_len, 2 * split_len),
    (2 * split_len, 4 * split_len),
    (4 * split_len, 6 * split_len),
    (6 * split_len, 8 * split_len),
    (8 * split_len, len_data),
]

eval_list = [
    (2 * split_len, 4 * split_len),
    (4 * split_len, 6 * split_len),
    (6 * split_len, 8 * split_len),
    (8 * split_len, len_data),
    (0 * split_len, 2 * split_len),
]

num_channel = 144
len_window = 513
print("5 fold cross-validation")
print(test_list)
print(eval_list)
# # train_list is remaining
for cross_id in range(5):
    test_container_X = np.zeros([3 * split_len, num_channel, len_window])
    test_container_Y = np.zeros(3 * split_len)
    test_count = 0
    eval_container_X = np.zeros([3 * split_len, num_channel, len_window])
    eval_container_Y = np.zeros(3 * split_len)
    eval_count = 0
    train_container_X = np.zeros([8 * split_len, num_channel, len_window])
    train_container_Y = np.zeros(8 * split_len)
    train_count = 0
    for i in range(len_data):
        if test_list[cross_id][0] < i and i < test_list[cross_id][1]:
            test_container_X[test_count, :, :] = all_data[i, :, :]
            test_container_Y[test_count] = all_label[i]
            test_count = test_count + 1
        elif eval_list[cross_id][0] < i and i < eval_list[cross_id][1]:
            eval_container_X[eval_count, :, :] = all_data[i, :, :]
            eval_container_Y[eval_count] = all_label[i]
            eval_count = eval_count + 1
        else:
            train_container_X[train_count, :, :] = all_data[i, :, :]
            train_container_Y[train_count] = all_label[i]
            train_count = train_count + 1
    print("Cross-{0}: Training Samples{1}, Validation Samples{2}, Testing Samples{3}".format(cross_id, train_count, eval_count, test_count))
    test_container_X = test_container_X[:test_count, :, :]
    eval_container_X = eval_container_X[:eval_count, :, :]
    train_container_X = train_container_X[:train_count, :, :]

    test_container_Y = test_container_Y[:test_count]
    eval_container_Y = eval_container_Y[:eval_count]
    train_container_Y = train_container_Y[:train_count]
    print("Save Test")
    print(test_container_X.shape)
    print(test_container_Y.shape)
    np.save(os.path.join(base_path, "test", "cross_{0}_".format(cross_id) + file_name_X),
            test_container_X)
    np.save(os.path.join(base_path, "test", "cross_{0}_".format(cross_id) + file_name_Y),
            test_container_Y)
    print("Save Val")
    print(eval_container_X.shape)
    print(eval_container_Y.shape)
    np.save(os.path.join(base_path, "eval", "cross_{0}_".format(cross_id) + file_name_X),
            eval_container_X)
    np.save(os.path.join(base_path, "eval", "cross_{0}_".format(cross_id) + file_name_Y),
            eval_container_Y)
    print("Save Training")
    print(train_container_X.shape)
    print(train_container_Y.shape)
    np.save(os.path.join(base_path, "train", "cross_{0}_".format(cross_id) + file_name_X),
            train_container_X)
    np.save(os.path.join(base_path, "train", "cross_{0}_".format(cross_id) + file_name_Y),
            train_container_Y)