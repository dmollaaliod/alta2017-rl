import matplotlib.pyplot as plt
import numpy as np
import time
import csv

with open('reinforce_log.csv') as f:
    csv_reader = csv.DictReader(f)
    lines = [f for f in csv_reader]

# episodes = [l['episode'] for l in lines]
# scores = [l['reward'] for l in lines]
# colors = [int(l['QID']) for l in lines]

averages = []
last1000 = []
for i, l in enumerate(lines):
    last1000i = int(l['episode'])
    last1000.append(float(l['reward']))
    if i < 1000:
        continue
    averages.append((last1000i, np.mean(last1000)))
    last1000 = last1000[1:]


with open('reinforce_eval.csv') as f:
    csv_reader = csv.DictReader(f)
    lines_eval = [f for f in csv_reader]

eval_episodes = set([int(l['episode']) for l in lines_eval])
eval_i = []
eval_results = []
len_test_data = 0
for e in sorted(eval_episodes):
    test_data = [float(l['reward']) for l in lines_eval
                           if int(l['episode']) == e]
    eval_result = np.mean(test_data)
    len_test_data = len(test_data)
    eval_i.append(e)
    eval_results.append(eval_result)
    
#plt.scatter(episodes, scores, alpha=0.1, c=colors)
ii, aa = zip(*averages)
plt.plot(ii, aa, color='black', label='Train')
plt.plot(eval_i, eval_results, color='red', label='Test')
plt.xlabel("Training Episodes")
plt.ylabel("ROUGE-L F1 Score")
plt.legend()
plt.show()
