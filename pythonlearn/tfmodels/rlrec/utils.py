import sys
import numpy as np
import os
from zoo.models.recommendation.utils import categorical_from_vocab_list

def custom_text_to_sequence(input_text,
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            split=" "):
    text = input_text.replace('\'s', '').lower()
    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    seq = text.split(split)
    return [w for w in seq if w]

def one_hot_encode(item, item_list):
    idx = categorical_from_vocab_list(item, item_list)
    encoded = [0 for _ in len(item_list)]
    encoded[idx] = 1
    return np.array(encoded)

def encode_user_embed(user_file ="./data/movielens/ml-1m/users.dat"):
    # feature length = 30
    gender_list = ['F','M']
    age_list = ["1", "18", "25", "35", "45", "50", "56"]
    income_list = [str(i) for i in range(21)]
    user_dict = {}
    with open(user_file) as user_f:
        for line in user_f:
            values = line.split("::")
            gender_one_hot = one_hot_encode(values[0], gender_list)
            age_one_hot = one_hot_encode(values[1], age_list)
            income_one_hot = one_hot_encode(values[2],income_list)
            embed = np.concatenate(gender_one_hot, age_one_hot, income_one_hot)
            user_dict[int(values[0])] = np.array(embed)
    return user_dict

cosine = lambda vA,vB:  np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
