import numpy
from tqdm import tqdm
import os
import sys
import random
import pickle

# Set path_prompt as argument for terminal
path_prompt = sys.argv[1]
number_sample = sys.argv[2]

# path_prompt = '/c/Users/NghiaPTIT/Downloads/grasp-anything/grasp-anything'
# path_prompt = "C:\\Users\\NghiaPTIT\\Downloads\\grasp-anything\\grasp-anything\\prompt"
prompts = os.listdir(path_prompt)

random.shuffle(prompts)

prompts = prompts[:number_sample]

with open("prompts.pkl", "wb") as f:
    pickle.dump(prompts, f)

vocab = set()

for name_prompt in tqdm(prompts):
    with open('../vdan/grasping/robotic-grasping/data/grasp-anything/prompt/' + name_prompt, 'rb') as f:
        prompt, objects = pickle.load(f)
        # Add object in objects to vocab, vocab is a set
        for obj in objects:
            vocab.add(obj)

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)