"""An environment for reinforcement learning"""
import os
import json
import rouge
from nltk import sent_tokenize

from xml_abstract_retriever import getAbstract

VERBOSE = 1
DEBUG = False

rouge_engine = rouge.Rouge()

def yield_candidate_text(questiondata, snippets_only=True):
    """Yield all candidate text for a question
    >>> data = json.load(open("BioASQ-trainingDataset5b.json", encoding='utf-8'))['questions']
    >>> y = yield_candidate_text(data[0], snippets_only=False)
    >>> next(y)
    ('15829955', 0, 'The identification of common variants that contribute to the genesis of human inherited disorders remains a significant challenge.')
    >>> next(y)
    ('15829955', 1, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes.')
    >>> y = yield_candidate_text(data[1], snippets_only=True)
    >>> next(y)
    ('55046d5ff8aee20f27000007', 0, 'the epidermal growth factor receptor (EGFR) ligands, such as epidermal growth factor (EGF) and amphiregulin (AREG)')
    >>> next(y)
    ('55046d5ff8aee20f27000007', 1, ' EGFR ligands epidermal growth factor (EGF), amphiregulin (AREG) and transforming growth factor alpha (TGFα)')
"""
    past_pubmed = set()
    sn_i = 0
    for sn in questiondata['snippets']:
        if snippets_only:
            for s in sent_tokenize(sn['text']):
                yield (questiondata['id'], sn_i, s)
                sn_i += 1
            continue

        pubmed_id = os.path.basename(sn['document'])
        if pubmed_id in past_pubmed:
            continue
        past_pubmed.add(pubmed_id)
        file_name = os.path.join("Task5bPubMed", pubmed_id+".xml")
        sent_i = 0
        for s in sent_tokenize(getAbstract(file_name, version="0")[0]):
            yield (pubmed_id, sent_i, s)
            sent_i += 1

class Environment:
    def __init__(self, jsonfile='BioASQ-trainingDataset5b.json'):
        if VERBOSE > 0:
            print("Starting reinforcement learning environment for data %s" % jsonfile)
        self.data = json.load(open(jsonfile, encoding='utf-8'))['questions']
        if DEBUG:
            print("Debugging mode in module rl")
            self.data = self.data[:10]

    def reset(self, qid):
        """Reset the environment using a specific query ID"""
        if VERBOSE > 0:
            print("Resetting environment to query ID %i" % qid)
        self.qid = qid
        self.id = self.data[qid]['id']
        self.qtype = self.data[qid]['type']
        self.question = self.data[qid]['body']
        self.candidates = [s[2] for s in yield_candidate_text(self.data[qid])]
        self.candidates = self.candidates[:30] # TODO: Remove the limit to first 30 candidates
        self.ideal_summaries = self.data[qid]['ideal_answer']
        if type(self.ideal_summaries) != list:
            self.ideal_summaries = [self.ideal_summaries]
        self.summary = list()
        self.actions = (0, 1)
        self.index = 0
        return {'summary': [],
                'next_candidate': 0,
                'done': len(self.candidates) == 0}

    def step(self, action):
        """Perform one action and observe result and reward"""
        assert action in self.actions
        assert self.index < len(self.candidates)

        if action == 1:
            self.summary.append(self.index)

        self.index += 1
        reward = 0
        done = self.index >= len(self.candidates)
        if done:
            summary_text = ' '.join([self.candidates[s] for s in self.summary])
            rouge_scores = [rouge_engine.get_scores(h, summary_text)[0] for h in self.ideal_summaries]
            #print(rouge_scores)
            rouge_l = max([r['rouge-l']['f'] for r in rouge_scores])
            reward = rouge_l

        return {'done': done,
                'reward': reward,
                'summary': self.summary,
                'next_candidate': self.index}

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import random
    env = Environment()
    env.reset(0)
    for i in range(len(env.candidates)):
        action = 0
        if random.random() > 0.8:
            action = 1
        state = env.step(action)
        print("Step %i; state =" % i, state)
