"""Using the REINFORCE algorithm"""
import numpy as np
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import json
import csv

from my_tokenizer import my_tokenize
from rl import Environment, yield_candidate_text, rouge_engine

VERBOSE = 1
RESTORE = False
CHECKPOINT_PATH = "reinforce.ckpt"
NN_CHECKPOINT_PATH = "nn_baseline.ckpt"
N_HIDDEN = 200
SAVE_EPISODES = 200
LOGFILE = "reinforce_log.csv"
NN_LOGFILE = "nn_baseline_log.csv"
EVALFILE = "reinforce_eval.csv"
NN_EVALFILE = "nn_baseline_eval.csv"
NOISE = 0.2 # Noise added when computing the action for training

def yieldRouge(CorpusFile, snippets_only=True):
    """yield ROUGE scores of all sentences in corpus
    >>> rouge = yieldRouge('BioASQ-trainingDataset5b.json')
    >>> target = (0, '15829955', 0, {'N-1': 0.1519, 'S4': 0.0, 'SU4': 0.04525, 'N-2': 0.0, 'L': 0.0}, 'The identification of common variants that contribute to the genesis of human inherited disorders remains a significant challenge.')
    >>> next(rouge) == target
    True
    >>> target2 = (0, '15829955', 1, {'N-1': 0.31915, 'S4': 0.02273, 'SU4': 0.09399, 'N-2': 0.13043, 'L': 0.04445}, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes.')
    >>> next(rouge) == target2
    True
    """
    data = json.load(open(CorpusFile, encoding='utf-8'))['questions']
    for qi in range(len(data)):
        if 'snippets' not in data[qi].keys():
            print("Warning: No snippets in question %s" % data[qi]['body'])
            continue
        ai = 0
        if type(data[qi]['ideal_answer']) == list:
            ideal_answers = data[qi]['ideal_answer']
        else:
            ideal_answers = [data[qi]['ideal_answer']]
        for (pubmedid, senti, sent) in yield_candidate_text(data[qi], snippets_only=snippets_only):                
            rouge_scores = [rouge_engine.get_scores(h, sent)[0] for h in ideal_answers]
            rouge_l = max([r['rouge-l']['f'] for r in rouge_scores])
            yield (qi, pubmedid, senti, rouge_l, sent)

def saveRouge(corpusfile, outfile, snippets_only=True):
    "Compute and save the ROUGE scores of the individual snippet sentences"
    with open(outfile,'w') as f:
        writer = csv.writer(f)
#        writer.writerow(('qid','snipid','sentid','N1','N2','L','S4','SU4','sentence text'))
        writer.writerow(('qid','pubmedid','sentid','L','sentence text'))
        for (qi,qsnipi,senti,F,sent) in yieldRouge(corpusfile, snippets_only=snippets_only):
            writer.writerow((qi,qsnipi,senti,F,sent))


class NNModel():
    def __init__(self, vocabulary_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X_state = tf.placeholder(tf.float32, shape=[None, 4*vocabulary_size]) # + 1])
            self.Q_state = tf.placeholder(tf.float32, shape=[None, vocabulary_size])
            self.episode = tf.placeholder(tf.float32)
            hidden = tf.layers.dense(tf.concat((self.X_state, self.Q_state), 1), N_HIDDEN, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            logits = tf.layers.dense(hidden, 1, activation=None)

            # The following code is from "Policy Gradients"
            # at https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
            self.outputs = tf.nn.sigmoid(logits)
            perturb = NOISE*3000/(3000+self.episode) # decrease the perturbation with each training episode
            p_left_and_right = tf.concat(axis=1, values=[(self.outputs+perturb)/(1+2*perturb), (1-self.outputs+perturb)/(1+2*perturb)])
            self.action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
            y = 1. - tf.to_float(self.action)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(cross_entropy)
            self.gradients = [grad for grad, _variable in grads_and_vars]
            self.gradient_placeholders = []
            grads_and_vars_feed = []
            for grad, variable in grads_and_vars:
                gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
                self.gradient_placeholders.append(gradient_placeholder)
                grads_and_vars_feed.append((gradient_placeholder, variable))
            self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

def NNbaseline(testfile=EVALFILE):
    """Evaluate a baseline that uses supervised NN"""
    KEEP_PROB = 0.2 # Rate of cells to keep at the dropout layer
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
    with open(NN_LOGFILE, 'w') as f:
        f.write("episode,reward,QID,summary\n")

    with open(NN_EVALFILE, 'w') as f:
        f.write("episode,reward,QID,summary\n")

    env = Environment(jsonfile='BioASQ-trainingDataset5b.json')
    all_data = env.data
    with open('rl-rouge5b.csv') as f:
        csvfile = csv.DictReader(f)
        all_rouge = [l for l in csvfile]

    if type(testfile) == None:
        all_indices = list(range(len(all_data)))
        np.random.shuffle(all_indices)
        split_boundary = int(len(all_indices)*.8)
        train_indices = all_indices[:split_boundary]
        test_indices = all_indices[split_boundary:]
    else:
        with open(testfile) as f:
            reader = csv.DictReader(f)
            test_indices = list(set(int(l['QID']) for l in reader) & set(range(len(all_data))))
        train_indices = [i for i in range(len(all_data)) if i not in test_indices]

    print("Train indices:", train_indices)
    print("Test indices:", test_indices)

    tfidf_train_text = [all_data[x]['body'] for x in train_indices]
    tfidf_train_text += [c[2] for x in train_indices for c in yield_candidate_text(all_data[x], snippets_only=True)]
    ideal_summaries_sentences = []
    for x in train_indices:
        ideal_summaries = all_data[x]['ideal_answer']
        if type(ideal_summaries) != list:
            ideal_summaries = [ideal_summaries]
        for ideal_sum in ideal_summaries:
            ideal_summaries_sentences += sent_tokenize(ideal_sum)
    tfidf_train_text += ideal_summaries_sentences
    #print(len(tfidf_train_text))
    #print(tfidf_train_text[:10])
    tfidf = TfidfVectorizer(tokenizer=my_tokenize)
    tfidf.fit(tfidf_train_text)
    vocabulary_size = len(tfidf.get_feature_names())

    graph = tf.Graph()
    with graph.as_default():
        X_state = tf.placeholder(tf.float32, shape=[None, 2*vocabulary_size]) # + 1])
        Q_state = tf.placeholder(tf.float32, shape=[None, vocabulary_size])
        Y_result = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32)
        dropout1 = tf.nn.dropout(tf.concat((X_state, Q_state), 1), keep_prob)
        hidden = tf.layers.dense(dropout1, N_HIDDEN, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        dropout2 = tf.nn.dropout(hidden, keep_prob)
        outputs = tf.layers.dense(dropout2, 1, activation=None)

        mse = tf.reduce_mean(tf.square(Y_result - outputs))
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(mse)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    if VERBOSE > 0:
        print("Training NN Baseline")
    with tf.Session(graph=graph) as sess:
        if RESTORE:
            saver.restore(sess, NN_CHECKPOINT_PATH)
        else:
            init.run()

        episode = 0
        while True:
            # 1. Train
            while True:
                train_x = np.random.choice(train_indices)
                observation = env.reset(train_x)
                if len(env.candidates) > 0:
                    break
            tfidf_all_candidates = tfidf.transform(env.candidates).todense()
            tfidf_all_text = tfidf.transform([" ".join(env.candidates)]).todense()[0,:]
            Y = [[float(l['L'])] for l in all_rouge if int(l['qid']) == env.qid][:len(env.candidates)]
            Q = np.tile(tfidf.transform([env.question]).todense()[0,:], (len(env.candidates),1))
            X = np.vstack([np.hstack([tfidf_all_text, c]) for c in tfidf_all_candidates])
            sess.run(train,
                    feed_dict={X_state: X,
                               Q_state: Q,
                               Y_result: Y,
                               keep_prob: KEEP_PROB})
            # 2. Evaluate
            predicted = sess.run(outputs,
                                 feed_dict={X_state: X,
                                            Q_state: Q,
                                            keep_prob: 1.0})
            n = nanswers[env.qtype]
            topn = sorted(predicted)[-n:]
            while not observation['done']:
                if predicted[observation['next_candidate']] >= topn[0]:
                    action = 1
                else:
                    action = 0
                observation = env.step(action)
            reward = observation['reward']
            print("Episode: %i, reward: %f" % (episode, reward))
            with open(NN_LOGFILE, 'a') as f:
                f.write('%i,%f,%i,"%s"\n' % (episode,reward,env.qid," ".join([str(x) for x in observation['summary']])))

            episode += 1
            if episode % SAVE_EPISODES == 0:
                print("Saving checkpoint in %s" % (NN_CHECKPOINT_PATH))
                saver.save(sess, NN_CHECKPOINT_PATH)
                # 3. Evaluate test data
                print("Testing results")
                test_results = []
                for test_x in test_indices:
                    observation = env.reset(test_x)
                    if len(env.candidates) == 0:
                        continue

                    tfidf_all_candidates = tfidf.transform(env.candidates).todense()
                    tfidf_all_text = tfidf.transform([" ".join(env.candidates)]).todense()[0,:]
                    Q = np.tile(tfidf.transform([env.question]).todense()[0,:], (len(env.candidates), 1))
                    X = np.vstack([np.hstack([tfidf_all_text, c]) for c in tfidf_all_candidates])
                    predicted = sess.run(outputs,
                                         feed_dict={X_state: X,
                                                    Q_state: Q,
                                                    keep_prob: 1.0})
                    n = nanswers[env.qtype]
                    topn = sorted(predicted)[-n:]
                    while not observation['done']:
                        if predicted[observation['next_candidate']] >= topn[0]:
                            action = 1
                        else:
                            action = 0
                        observation = env.step(action)
                    reward = observation['reward']


                    test_results.append(reward)
                    with open(NN_EVALFILE, 'a') as f:
                            f.write('%i,%f,%i,"%s"\n' % (episode,reward,env.qid," ".join([str(x) for x in observation['summary']])))
                print("Mean of evaluation results:", np.mean(test_results))


def baseline(testfile=EVALFILE):
    """Evaluate a baseline that returns the first n sentences"""
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
    env = Environment(jsonfile='BioASQ-trainingDataset5b.json')
    if type(testfile) == None:
        alldata = list(range(len(env.data)))
        np.random.shuffle(alldata)
        split_boundary = int(len(alldata)*.8)
        train_indices = alldata[:split_boundary]
        test_indices = alldata[split_boundary:]
    else:
        with open(testfile) as f:
            reader = csv.DictReader(f)
            test_indices = list(set(int(l['QID']) for l in reader))
        
    scores = []
    for x in test_indices:
        observation = env.reset(x)
        n = nanswers[env.qtype]
        if len(env.candidates) == 0:
            continue

        while not observation['done']:
            this_candidate = observation['next_candidate']
            if this_candidate < n:
                action = 1
            else:
                action = 0
            observation = env.step(action)
        reward = observation['reward']
        scores.append(reward)
    return np.mean(scores)

def train():

    with open(LOGFILE, 'w') as f:
        f.write("episode,reward,QID,summary\n")

    with open(EVALFILE, 'w') as f:
        f.write("episode,reward,QID,summary\n")

    env = Environment(jsonfile='BioASQ-trainingDataset5b.json')
    alldata = list(range(len(env.data)))
    np.random.shuffle(alldata)
    split_boundary = int(len(alldata)*.8)
    train_indices = alldata[:split_boundary]
    test_indices = alldata[split_boundary:]

    # train tf.idf
    if VERBOSE > 0:
        print("Training tf.idf")
    tfidf_train_text = [env.data[x]['body'] for x in train_indices]
    tfidf_train_text += [c[2] for x in train_indices for c in yield_candidate_text(env.data[x], snippets_only=True)]
    ideal_summaries_sentences = []
    for x in train_indices:
        ideal_summaries = env.data[x]['ideal_answer']
        if type(ideal_summaries) != list:
            ideal_summaries = [ideal_summaries]
        for ideal_sum in ideal_summaries:
            ideal_summaries_sentences += sent_tokenize(ideal_sum)
    tfidf_train_text += ideal_summaries_sentences
    #print(len(tfidf_train_text))
    #print(tfidf_train_text[:10])
    tfidf = TfidfVectorizer(tokenizer=my_tokenize)
    tfidf.fit(tfidf_train_text)
    nnModel = NNModel(len(tfidf.get_feature_names()))

    if VERBOSE > 0:
        print("Training REINFORCE")
    with tf.Session(graph=nnModel.graph) as sess:
        if RESTORE:
            nnModel.saver.restore(sess, CHECKPOINT_PATH)
        else:
            nnModel.init.run()

        while True:
            train_x = np.random.choice(train_indices)
            observation = env.reset(train_x) # Reset to a random question
            if len(env.candidates) > 0:
                break
        tfidf_all_candidates = tfidf.transform(env.candidates)
        tfidf_all_text = tfidf.transform([" ".join(env.candidates)]).todense()[0,:]

        all_gradients = []
        episode = 0
        while True:
            # The following code is based on "Policy Gradients"
            # at https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
            this_candidate = observation['next_candidate']
            tfidf_this_candidate = tfidf_all_candidates[this_candidate].todense()
            tfidf_remaining_candidates = tfidf.transform([" ".join(env.candidates[this_candidate + 1:])]).todense()[0,:]
            tfidf_summary = tfidf.transform([" ".join([env.candidates[x] for x in observation['summary']])]).todense()[0,:]
            tfidf_question = tfidf.transform([env.question]).todense()[0,:]
            #print(tfidf_question.shape)
            XState = np.hstack([tfidf_all_text, tfidf_this_candidate, tfidf_remaining_candidates, tfidf_summary]) #, [[len(observation['summary'])]]])
            action_val, gradients_val = sess.run([nnModel.action, nnModel.gradients],
                                                  feed_dict={nnModel.X_state: XState,
                                                             nnModel.Q_state: tfidf_question,
                                                             nnModel.episode: episode})
            all_gradients.append(gradients_val)
            #action = 1 if np.random.uniform() < action_prob else 0
            observation = env.step(action_val)

            if observation['done']:
                # reward all actions that lead to the summary
                reward = observation['reward']
                print("Episode: %i, reward: %f" % (episode, reward))
                with open(LOGFILE, 'a') as f:
                    f.write('%i,%f,%i,"%s"\n' % (episode,reward,env.qid," ".join([str(x) for x in observation['summary']])))

                feed_dict = {}
                #print(nnModel.gradient_placeholders[0].shape)
                for var_index, grad_placeholder in enumerate(nnModel.gradient_placeholders):
                    mean_gradients = np.mean(
                        [reward * one_gradient[var_index]
                         for one_gradient in all_gradients],
                        axis=0
                    )
                    feed_dict[grad_placeholder] = mean_gradients
                sess.run(nnModel.training_op, feed_dict=feed_dict)

                episode += 1
                if episode % SAVE_EPISODES == 0:
                    print("Saving checkpoint in %s" % (CHECKPOINT_PATH))
                    nnModel.saver.save(sess, CHECKPOINT_PATH)
                    print("Testing results")
                    test_results = []
                    for test_x in test_indices:
                        observation = env.reset(test_x)
                        if len(env.candidates) == 0:
                            continue
                        
                        tfidf_all_candidates = tfidf.transform(env.candidates)
                        tfidf_all_text = tfidf.transform([" ".join(env.candidates)]).todense()[0,:]
                        while not observation['done']:
                            this_candidate = observation['next_candidate']
                            tfidf_this_candidate = tfidf_all_candidates[this_candidate].todense()
                            tfidf_remaining_candidates = tfidf.transform([" ".join(env.candidates[this_candidate + 1:])]).todense()[0,:]
                            tfidf_summary = tfidf.transform([" ".join([env.candidates[x] for x in observation['summary']])]).todense()[0,:]
                            tfidf_question = tfidf.transform([env.question]).todense()[0,:]
                            #print(tfidf_question.shape)
                            XState = np.hstack([tfidf_all_text, tfidf_this_candidate, tfidf_remaining_candidates, tfidf_summary]) #, [[len(observation['summary'])]]])
                            output_val = sess.run(nnModel.outputs,
                                                  feed_dict={nnModel.X_state: XState,
                                                             nnModel.Q_state: tfidf_question})
                            action_val = 0
                            if output_val < 0.5:
                                action_val = 1
                            observation = env.step(action_val)
                        reward = observation['reward']
                        test_results.append(reward)
                        with open(EVALFILE, 'a') as f:
                            f.write('%i,%f,%i,"%s"\n' % (episode,reward,env.qid," ".join([str(x) for x in observation['summary']])))
                    print("Mean of evaluation results:", np.mean(test_results))

                # Pick next training question
                while True:
                    train_x = np.random.choice(train_indices)
                    observation = env.reset(train_x) # Reset to a random question
                    if len(env.candidates) > 0:
                        break
                all_gradients = []
                tfidf_all_candidates = tfidf.transform(env.candidates)
                tfidf_all_text = tfidf.transform([" ".join(env.candidates)]).todense()[0,:]
                    

if __name__ == "__main__":
    train()
