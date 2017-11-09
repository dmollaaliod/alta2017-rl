# bioasq-rl
Experiments using Reinforcement Learning for BioASQ as described in
the following paper:

D. Molla (2017). Towards the use of deep reinforce learning with global policy
for query-based extractive summarisation. *Proceedings ALTA 2017*, Brisbane, 
Australia.

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

## Run "first n" baseline
```python
$ python3
>>> from rl import reinforce
>>> reinforce.baseline()
```
