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
--lr=2e-3 (learning rate)
--batch_size=50 (number of images in a batch)
--num_rounds=20000 (number of total training rounds)
--num_games_per_round=20 (number of games per round)
--vocab_size=5 (vocabulary size)
--max_sentence_len=20 (maximum sentence length)
--data_n_samples=100 (number of samples per color, shape combination)

```
