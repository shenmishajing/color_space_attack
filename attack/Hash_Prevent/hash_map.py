import random

forward_map = {}
backward_map = {}

f = list(range(256))
b = list(range(256))
random.shuffle(f)
for f,b in zip(f,b):
    forward_map[f]=b
    backward_map[b]=f
    





