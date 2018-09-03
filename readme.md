# Compositional Obverter Communication Learning From Raw Visual Input - Pytorch Implementation


[https://arxiv.org/abs/1804.02341](https://arxiv.org/abs/1804.02341)

### Dataset


#### Preparing the dataset


```
$ tar -xzvf assets/dataset.tar.gz -C assets
```

#### (Optional) Creating the dataset from scratch
##### Requirements

* python 3.6
* [povray](http://www.povray.org/download/)
* [vapory](https://github.com/Zulko/vapory)

##### Running

```
$ python create_ds.py
```

This will create all images in the dataset: 8 colors (red, blue, green, white, gray, yellow, cyan, magenta) x 5 shapes (box, sphere, cylinder, torus (donut), ellipsoid) x 100 samples in different locations and angles. 

![yellow-box-2](https://user-images.githubusercontent.com/2988446/44865469-0e96de00-ac8b-11e8-9580-0818d2c8d52e.png)
![red-torus-0](https://user-images.githubusercontent.com/2988446/44865489-21111780-ac8b-11e8-988f-02a39727eac4.png)
![blue-sphere-1](https://user-images.githubusercontent.com/2988446/44865509-30906080-ac8b-11e8-9204-09a4aebaf3f0.png)
![green-cylinder-2](https://user-images.githubusercontent.com/2988446/44865520-3ede7c80-ac8b-11e8-9ef5-40f8721b858d.png)
![gray-ellipsoid-0](https://user-images.githubusercontent.com/2988446/44865543-49991180-ac8b-11e8-994e-a07c10d64b0f.png)

Command line options:

```
--n_samples=100
--seed=0
```

### Training


#### Requirements

* python 3.6
* [pytorch](https://pytorch.org/) == 0.4
* matplotlib

#### Running

```
$ python train.py
```

Command line options:

```
--lr=6e-4 (learning rate)
--batch_size=50 (number of images in a batch)
--num_rounds=20000 (number of total training rounds)
--num_games_per_round=20 (number of games per round)
--vocab_size=5 (vocabulary size)
--max_sentence_len=20 (maximum sentence length)
--data_n_samples=100 (number of samples per color, shape combination)

```

### Results

Output sample from round #9702

```
...
message: 'ec', speaker object: ('ellipsoid', 'red'), speaker score: 0.96, listener object: ('ellipsoid', 'red'), label: 1, listener score: 0.89
message: 'ec', speaker object: ('box', 'blue'), speaker score: 0.96, listener object: ('box', 'blue'), label: 1, listener score: 0.97
message: 'cbe', speaker object: ('torus', 'magenta'), speaker score: 0.96, listener object: ('torus', 'magenta'), label: 1, listener score: 0.96
message: 'ecb', speaker object: ('sphere', 'magenta'), speaker score: 0.97, listener object: ('sphere', 'magenta'), label: 1, listener score: 0.97
message: 'ea', speaker object: ('box', 'white'), speaker score: 0.96, listener object: ('box', 'white'), label: 1, listener score: 0.96
message: 'ee', speaker object: ('sphere', 'cyan'), speaker score: 0.96, listener object: ('sphere', 'cyan'), label: 1, listener score: 0.97
message: 'eeb', speaker object: ('box', 'gray'), speaker score: 0.97, listener object: ('box', 'gray'), label: 1, listener score: 0.97
message: 'ecac', speaker object: ('box', 'cyan'), speaker score: 0.97, listener object: ('box', 'cyan'), label: 1, listener score: 0.95
message: 'ecc', speaker object: ('torus', 'green'), speaker score: 0.97, listener object: ('torus', 'green'), label: 1, listener score: 0.95
message: 'bed', speaker object: ('ellipsoid', 'cyan'), speaker score: 0.96, listener object: ('ellipsoid', 'cyan'), label: 1, listener score: 0.73
message: 'b', speaker object: ('sphere', 'white'), speaker score: 0.97, listener object: ('sphere', 'white'), label: 1, listener score: 0.97
message: 'bee', speaker object: ('torus', 'white'), speaker score: 0.97, listener object: ('torus', 'white'), label: 1, listener score: 0.92
message: 'cdb', speaker object: ('box', 'magenta'), speaker score: 0.96, listener object: ('box', 'yellow'), label: 0, listener score: 0.00
message: 'ec', speaker object: ('torus', 'green'), speaker score: 0.97, listener object: ('torus', 'blue'), label: 0, listener score: 0.00
message: 'ebe', speaker object: ('ellipsoid', 'yellow'), speaker score: 0.97, listener object: ('ellipsoid', 'green'), label: 0, listener score: 0.00
message: 'cd', speaker object: ('torus', 'red'), speaker score: 0.95, listener object: ('torus', 'cyan'), label: 0, listener score: 0.00
message: 'cb', speaker object: ('sphere', 'red'), speaker score: 0.96, listener object: ('sphere', 'blue'), label: 0, listener score: 0.00
message: 'ebc', speaker object: ('cylinder', 'white'), speaker score: 0.96, listener object: ('cylinder', 'cyan'), label: 0, listener score: 0.00
message: 'd', speaker object: ('ellipsoid', 'white'), speaker score: 0.99, listener object: ('ellipsoid', 'blue'), label: 0, listener score: 0.00
message: 'cda', speaker object: ('torus', 'red'), speaker score: 0.96, listener object: ('torus', 'white'), label: 0, listener score: 0.00
message: 'ebc', speaker object: ('cylinder', 'white'), speaker score: 0.96, listener object: ('cylinder', 'gray'), label: 0, listener score: 0.07
message: 'ecb', speaker object: ('box', 'blue'), speaker score: 0.96, listener object: ('box', 'white'), label: 0, listener score: 0.00
message: 'eeb', speaker object: ('ellipsoid', 'green'), speaker score: 0.97, listener object: ('ellipsoid', 'magenta'), label: 0, listener score: 0.00
message: 'ecac', speaker object: ('box', 'cyan'), speaker score: 0.97, listener object: ('box', 'white'), label: 0, listener score: 0.00
message: 'ebc', speaker object: ('sphere', 'blue'), speaker score: 0.97, listener object: ('sphere', 'white'), label: 0, listener score: 0.00
message: 'ed', speaker object: ('torus', 'gray'), speaker score: 0.95, listener object: ('torus', 'red'), label: 0, listener score: 0.00
message: 'cc', speaker object: ('cylinder', 'red'), speaker score: 1.00, listener object: ('cylinder', 'blue'), label: 0, listener score: 0.00
message: 'ede', speaker object: ('ellipsoid', 'blue'), speaker score: 0.96, listener object: ('sphere', 'blue'), label: 0, listener score: 0.00
message: 'ce', speaker object: ('torus', 'yellow'), speaker score: 0.97, listener object: ('sphere', 'yellow'), label: 0, listener score: 0.00
message: 'cda', speaker object: ('torus', 'red'), speaker score: 0.96, listener object: ('cylinder', 'red'), label: 0, listener score: 0.00
message: 'cdb', speaker object: ('box', 'magenta'), speaker score: 0.96, listener object: ('torus', 'magenta'), label: 0, listener score: 0.00
message: 'ebe', speaker object: ('cylinder', 'gray'), speaker score: 0.97, listener object: ('ellipsoid', 'gray'), label: 0, listener score: 0.00
message: 'ce', speaker object: ('torus', 'yellow'), speaker score: 0.96, listener object: ('sphere', 'yellow'), label: 0, listener score: 0.00
message: 'ccc', speaker object: ('cylinder', 'magenta'), speaker score: 1.00, listener object: ('box', 'magenta'), label: 0, listener score: 0.00
message: 'eb', speaker object: ('torus', 'cyan'), speaker score: 0.96, listener object: ('ellipsoid', 'cyan'), label: 0, listener score: 0.00
message: 'beb', speaker object: ('sphere', 'gray'), speaker score: 0.97, listener object: ('torus', 'gray'), label: 0, listener score: 0.00
message: 'cec', speaker object: ('cylinder', 'cyan'), speaker score: 0.98, listener object: ('sphere', 'cyan'), label: 0, listener score: 0.00
message: 'cdb', speaker object: ('box', 'magenta'), speaker score: 0.96, listener object: ('sphere', 'green'), label: 0, listener score: 0.00
message: 'ebc', speaker object: ('sphere', 'blue'), speaker score: 0.97, listener object: ('box', 'red'), label: 0, listener score: 0.00
message: 'ebd', speaker object: ('torus', 'blue'), speaker score: 0.95, listener object: ('sphere', 'green'), label: 0, listener score: 0.67
message: 'eebb', speaker object: ('ellipsoid', 'green'), speaker score: 0.97, listener object: ('ellipsoid', 'green'), label: 1, listener score: 0.96
message: 'ccd', speaker object: ('box', 'red'), speaker score: 0.98, listener object: ('cylinder', 'red'), label: 0, listener score: 0.00
message: 'b', speaker object: ('sphere', 'white'), speaker score: 0.97, listener object: ('cylinder', 'green'), label: 0, listener score: 0.00
message: 'c', speaker object: ('box', 'red'), speaker score: 0.95, listener object: ('sphere', 'red'), label: 0, listener score: 0.00
message: 'b', speaker object: ('ellipsoid', 'gray'), speaker score: 0.96, listener object: ('torus', 'red'), label: 0, listener score: 0.00
message: 'ebb', speaker object: ('sphere', 'green'), speaker score: 0.95, listener object: ('box', 'white'), label: 0, listener score: 0.00
message: 'ea', speaker object: ('box', 'white'), speaker score: 0.96, listener object: ('torus', 'white'), label: 0, listener score: 0.00
message: 'ed', speaker object: ('torus', 'gray'), speaker score: 0.96, listener object: ('box', 'white'), label: 0, listener score: 0.92
message: 'bee', speaker object: ('sphere', 'gray'), speaker score: 0.97, listener object: ('cylinder', 'white'), label: 0, listener score: 0.00
message: 'ed', speaker object: ('torus', 'gray'), speaker score: 0.96, listener object: ('cylinder', 'gray'), label: 0, listener score: 0.00
batch accuracy 0.96
batch loss 0.09183049947023392
*******
Round average accuracy: 96.90
Round average sentence length: 2.6
Round average loss: 0.1
```

![graph](https://user-images.githubusercontent.com/2988446/44970524-cb728e80-af5a-11e8-8ee4-49fa1034917a.png)


### Differences from paper

This repository was written with the intention to as close as possible to the paper's described methods.

Two differences are known:

* This implementation contains less convolution layers, and less filters in each layer
* This dataset is using the torus (donut) shape instead of the paper's capsule