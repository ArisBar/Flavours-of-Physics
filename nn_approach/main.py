
from loss import accuracy, crossent
from data import Data
import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
import lasagne

d = Data()

srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


def init_params():

    params = {}

    m = 46

    params['W1'] = theano.shared(0.02 * rng.normal(size = (m,2*2048)).astype('float32'))
    params['W2'] = theano.shared(0.02 * rng.normal(size = (2*2048,2*2048)).astype('float32'))
    params['W3'] = theano.shared(0.02 * rng.normal(size = (2*2048,2)).astype('float32'))

    params['b1'] = theano.shared(np.zeros(shape = (2*2048,)).astype('float32'))
    params['b2'] = theano.shared(np.zeros(shape = (2*2048,)).astype('float32'))
    params['b3'] = theano.shared(np.zeros(shape = (2,)).astype('float32'))

    return params

def bn(inp):
    return (inp - inp.mean(axis = 0, keepdims=True)) / (0.001 + inp.std(axis = 0, keepdims = True))

def network(params, x,y,p1,p2):

    #x *= srng.binomial(n=1,p=p1,size=x.shape,dtype='float32').astype('float32')/p1

    h1 = T.nnet.relu(bn(T.dot(bn(x), params['W1']) + params['b1']))
    #h1 *= srng.binomial(n=1,p=p2,size=h1.shape,dtype='float32').astype('float32')/p2
    h2 = T.nnet.relu(bn(T.dot(h1, params['W2']) + params['b2']))
    #h2 *= srng.binomial(n=1,p=p2,size=h2.shape,dtype='float32').astype('float32')/p2
    h3 = bn(T.dot(h2, params['W3']) + params['b3'])

    p = T.nnet.softmax(h3)

    loss = crossent(p, y)
    acc = accuracy(p,y)

    return {'loss' : loss, 'p' : p, 'acc' : acc}

params = init_params()

x = T.matrix()
y = T.ivector()

out_train = network(params,x,y,1.0,1.0)
out_valid = network(params,x,y,1.0,1.0)

adv_grad = T.grad(out_train['loss'], x)
adv_x = x + 0.1 * (adv_grad/T.abs_(adv_grad))

updates = lasagne.updates.adam(out_train['loss'], params.values())

train_method = theano.function([x,y], outputs = {'loss' : out_train['loss'], 'p' : out_train['p'], 'acc' : out_train['acc'], 'adv_x' : adv_x}, updates = updates)

valid_method = theano.function([x,y], outputs = {'loss' : out_valid['loss'], 'p' : out_valid['p'], 'acc' : out_valid['acc']})

for iteration in xrange(0,100000):
    x,y = d.get(mb=256)

    y = y.flatten()
    out1 = train_method(x,y)

    out2 = train_method(out1['adv_x'],y)

    if iteration % 2000 == 0:

        print "orig acc", out1['acc']
        print "adv acc", out2['acc']

        print iteration

        x,y = d.get(mb=4096, segment='train')
        y = y.flatten()
        out = valid_method(x,y)


        print "train", out['acc']
        
        x,y = d.get(mb=4096, segment='valid')
        y = y.flatten()
        out = valid_method(x,y)
        print "Validation", out['acc']

        

