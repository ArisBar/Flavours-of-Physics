import numpy as np
import random

class Data:

    def __init__(self):
        train_fh = open("/u/lambalex/data/physics/training.csv", "r")
        #valid_fh = open("/u/lambalex/data/physics/valid_split.csv", "r")

        line = train_fh.readline().rstrip("\n")

        lst = line.split(",")

        labelInd = [lst.index("signal")]
        blockInd = [lst.index("id"), lst.index("min_ANNmuon"), lst.index("production"), lst.index("mass")]
        allInd = range(len(lst))

        feats = []
        labels = []

        for line in train_fh:
            lst = line.rstrip("\n").split(",")
            feat = []
            label = []

            for i in range(len(lst)):
                if i in labelInd:
                    label.append(int(lst[i]))
                elif i in blockInd:
                    pass
                else:
                    feat.append(float(lst[i]))

            feat = np.asarray([feat])
            label = np.asarray([label])

            feats.append(feat)
            labels.append(label)



        featTrain = np.vstack(feats).astype('float32')
        labelTrain = np.vstack(labels).astype('int32')

        print "shape", featTrain.shape

        p = np.random.permutation(featTrain.shape[0])

        featTrain = featTrain[p]
        labelTrain = labelTrain[p]


        featNormTrain = np.log(featTrain - np.min(featTrain, axis = 0) + 1.0)


        self.trainx = featNormTrain
        self.trainy = labelTrain

    def get(self,mb=32,segment='train'):
        assert segment in ["train", "valid"]

        if segment == "train":
            r = random.randint(500 + mb, self.trainx.shape[0] - mb)
        elif segment == 'valid':
            r = random.randint(0, 500)

        return (self.trainx[r:r+mb],self.trainy[r:r+mb])

    def process(self):
        pass

if __name__ == "__main__":
    d = Data()

    print d.get()


