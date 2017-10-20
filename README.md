# bioasq-rl
Experiments using Reinforcement Learning for BioASQ

## Train and evaluate reinforce for debugging

```python
$ python3
>>> import rl
>>> rl.DEBUG = True
>>> from rl import reinforce
>>> reinforce.train()
```

## Train and evaluate reinforce for production
```python
>>> from rl import reinforce
>>> reinforce.train()
```

## Train and evaluate NNbaseline for debugging
```python
$ python3
>>> import rl
>>> rl.DEBUG = True
>>> from rl import reinforce
>>> reinforce.NNbaseline()
```

## Train and evaluate NNbaseline for production
```python
$ python3
>>> from rl import reinforce
>>> reinforce.NNbaseline()
```

## Run "first n" baseline
```python
$ python3
>>> from rl import reinforce
>>> reinforce.baseline()
```
