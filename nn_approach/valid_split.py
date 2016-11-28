import sys
import random

first = True

p = 0.8

train_file = open("/u/lambalex/data/physics/train_split.csv", "w")
valid_file = open("/u/lambalex/data/physics/valid_split.csv", "w")

for line in sys.stdin: 

    if first:
        train_file.write(line)
        valid_file.write(line)
    else:
        if random.uniform(0,1) < p:
            train_file.write(line)
        else:
            valid_file.write(line)

    first = False

