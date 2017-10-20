"""report_eval.py - Print train and test evaluation results"""
import csv
import numpy as np

LOGFILE = "../reinforce_log.csv"
EVALFILE = "../reinforce_eval.csv"

with open(LOGFILE) as f:
    reader = csv.DictReader(f)
    loglines = [l for l in reader]

with open(EVALFILE) as f:
    reader = csv.DictReader(f)
    evallines = [l for l in reader]

print("| Episodes | Train ROUGE F1 | Test ROUGE F1 |")
print("|----------+----------------+---------------|")
    
for episode in (1000, 5000, 10000, 20000, 50000, 100000):
    lines = [float(l['reward']) for l in loglines[episode-1000:episode]]
    trainscore = np.mean(lines)

    lines = [float(l['reward']) for l in evallines if int(l['episode']) == episode]
    testscore = np.mean(lines)

    print("| %i | %f | %f |" % (episode, trainscore, testscore))
