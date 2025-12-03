import os
import random
import time
import warnings
import concurrent.futures
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from nltk.corpus import wordnet as wn
from nb import NaiveBayesWSD, load_data, simplify_pos
from svm import SVMWSD
from baseline import MostFrequentSenseWSD
from lesk import simple_lesk, extended_lesk, TfidfLesk, EmbeddingLesk

warnings.filterwarnings("ignore")

def get_wnsn_from_synset(lemma, synset):
    if not synset:
        return None
    
    possible_synsets = wn.synsets(lemma)
    
    try:
        index = possible_synsets.index(synset)
        return str(index + 1)
    except ValueError:
        return None

def evaluate_model(name, predictions, true_labels):
    # Return metrics dict instead of printing immediately to avoid race conditions
    assert len(predictions) == len(true_labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        "Name": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def run_lesk_evaluation(sentences, method='simple'):
    predictions = []
    true_labels = []
    
    for sent in sentences:
        # Construct context sentence (list of words)
        context_sentence = [t['word'] for t in sent if t['word']]
        
        for i, token in enumerate(sent):
            lemma = token['lemma']
            true_sense = token['sense']
            pos = simplify_pos(token['pos'])
            
            if lemma and lemma != 'None' and true_sense and pos:
                # Run Lesk
                if method == 'simple':
                    synset = simple_lesk(context_sentence, lemma, pos=pos)
                else:
                    synset = extended_lesk(context_sentence, lemma, pos=pos)
                
                pred_sense = get_wnsn_from_synset(lemma, synset)
                
                if pred_sense is None:
                    pred_sense = "0" 
                
                predictions.append(pred_sense)
                true_labels.append(true_sense)
                    
    return predictions, true_labels

def run_model_evaluation(model, sentences):
    predictions = []
    true_labels = []
    
    for sent in sentences:
        for i, token in enumerate(sent):
            lemma = token['lemma']
            true_sense = token['sense']
            pos = simplify_pos(token['pos'])
            
            if lemma and lemma != 'None' and true_sense and pos:
                pred_sense = model.predict(sent, i)
                
                if pred_sense is None:
                    pred_sense = "0"
                
                predictions.append(pred_sense)
                true_labels.append(true_sense)
                
    return predictions, true_labels

# Wrapper functions for parallel execution
def task_nb_pos(train_data, eval_data):
    print("Starting Naive Bayes (with POS)...")
    model = NaiveBayesWSD(use_pos=True)
    model.train(train_data)
    preds, true = run_model_evaluation(model, eval_data)
    return evaluate_model("Naive Bayes (with POS)", preds, true)

def task_nb_no_pos(train_data, eval_data):
    print("Starting Naive Bayes (without POS)...")
    model = NaiveBayesWSD(use_pos=False)
    model.train(train_data)
    preds, true = run_model_evaluation(model, eval_data)
    return evaluate_model("Naive Bayes (without POS)", preds, true)

def task_svm_pos(train_data, eval_data):
    print("Starting SVM (with POS)...")
    model = SVMWSD(use_pos=True)
    model.train(train_data)
    preds, true = run_model_evaluation(model, eval_data)
    return evaluate_model("SVM (with POS)", preds, true)

def task_svm_no_pos(train_data, eval_data):
    print("Starting SVM (without POS)...")
    model = SVMWSD(use_pos=False)
    model.train(train_data)
    preds, true = run_model_evaluation(model, eval_data)
    return evaluate_model("SVM (without POS)", preds, true)

def task_lesk_simple(eval_data):
    print("Starting Simple Lesk...")
    preds, true = run_lesk_evaluation(eval_data, method='simple')
    return evaluate_model("Simple Lesk", preds, true)

def task_lesk_extended(eval_data):
    print("Starting Extended Lesk...")
    preds, true = run_lesk_evaluation(eval_data, method='extended')
    return evaluate_model("Extended Lesk", preds, true)

def task_lesk_tfidf(train_data, eval_data):
    print("Starting TF-IDF Lesk...")
    # Initialize and train TF-IDF
    tfidf = TfidfLesk()
    tfidf.fit(train_data)
    
    predictions = []
    true_labels = []
    
    for sent in eval_data:
        context_sentence = [t['word'] for t in sent if t['word']]
        for i, token in enumerate(sent):
            lemma = token['lemma']
            true_sense = token['sense']
            pos = simplify_pos(token['pos'])
            
            if lemma and lemma != 'None' and true_sense and pos:
                synset = tfidf.predict(context_sentence, lemma, pos=pos)
                pred_sense = get_wnsn_from_synset(lemma, synset)
                if pred_sense is None: pred_sense = "0"
                predictions.append(pred_sense)
                true_labels.append(true_sense)
                
    return evaluate_model("TF-IDF Lesk", predictions, true_labels)

def task_lesk_embedding(eval_data):
    print("Starting Embedding Lesk...")
    # Initialize Embedding Lesk (loads model)
    emb_lesk = EmbeddingLesk()
    
    predictions = []
    true_labels = []
    
    for sent in eval_data:
        context_sentence = [t['word'] for t in sent if t['word']]
        for i, token in enumerate(sent):
            lemma = token['lemma']
            true_sense = token['sense']
            pos = simplify_pos(token['pos'])
            
            if lemma and lemma != 'None' and true_sense and pos:
                synset = emb_lesk.predict(context_sentence, lemma, pos=pos)
                pred_sense = get_wnsn_from_synset(lemma, synset)
                if pred_sense is None: pred_sense = "0"
                predictions.append(pred_sense)
                true_labels.append(true_sense)
                
    return evaluate_model("Embedding Lesk", predictions, true_labels)

def task_mfs(train_data, eval_data):
    print("Starting Most Frequent Sense (MFS)...")
    model = MostFrequentSenseWSD(use_pos=True)
    model.train(train_data)
    preds, true = run_model_evaluation(model, eval_data)
    return evaluate_model("Most Frequent Sense (MFS)", preds, true)

def run_learning_curve(train_data, eval_data):
    print("\n=== Learning Curve Analysis ===")
    fractions = [0.1, 0.25, 0.5, 1.0]
    
    results = {
        "NB": [],
        "SVM": []
    }
    
    for frac in fractions:
        print(f"\n--- Training with {frac*100}% of data ---")
        # Subset training data
        subset_size = int(len(train_data) * frac)
        train_subset = train_data[:subset_size]
        print(f"Training set size: {len(train_subset)} sentences")
        
        # Train and evaluate NB
        nb = NaiveBayesWSD(use_pos=True)
        nb.train(train_subset)
        nb_preds, nb_true = run_model_evaluation(nb, eval_data)
        nb_acc = accuracy_score(nb_true, nb_preds)
        results["NB"].append(nb_acc)
        print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
        
        # Train and evaluate SVM
        svm = SVMWSD(use_pos=True)
        svm.train(train_subset)
        svm_preds, svm_true = run_model_evaluation(svm, eval_data)
        svm_acc = accuracy_score(svm_true, svm_preds)
        results["SVM"].append(svm_acc)
        print(f"SVM Accuracy:         {svm_acc:.4f}")
        
    print("\n--- Learning Curve Summary ---")
    print(f"{'Fraction':<10} {'NB Accuracy':<15} {'SVM Accuracy':<15}")
    for i, frac in enumerate(fractions):
        print(f"{frac:<10} {results['NB'][i]:<15.4f} {results['SVM'][i]:<15.4f}")

def main():
    data_folder = "data"
    if not os.path.exists(data_folder):
        data_folder = os.path.join(os.path.dirname(__file__), "data")
        
    print(f"Loading data from {data_folder}...")
    all_sentences = load_data(data_folder)
    print(f"Loaded {len(all_sentences)} sentences.")
    
    # Split data
    random.seed(42)
    random.shuffle(all_sentences)
    split_idx = int(len(all_sentences) * 0.8)
    train_data = all_sentences[:split_idx]
    test_data = all_sentences[split_idx:]
    
    # Use a subset of test data for evaluation to be fast
    eval_data = test_data[:500]
    print(f"Using {len(eval_data)} sentences for evaluation.")

    # Run evaluations in parallel
    print("\nStarting parallel evaluation of models...")
    start_time = time.time()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(task_nb_pos, train_data, eval_data),
            executor.submit(task_nb_no_pos, train_data, eval_data),
            executor.submit(task_svm_pos, train_data, eval_data),
            executor.submit(task_svm_no_pos, train_data, eval_data),
            executor.submit(task_lesk_simple, eval_data),
            executor.submit(task_lesk_extended, eval_data),
            executor.submit(task_lesk_tfidf, train_data, eval_data),
            executor.submit(task_lesk_embedding, eval_data),
            executor.submit(task_mfs, train_data, eval_data)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Task failed with exception: {e}")

    print(f"\nAll evaluations finished in {time.time() - start_time:.2f}s")
    
    # Print results sorted by name
    results.sort(key=lambda x: x['Name'])
    for res in results:
        print(f"\n--- {res['Name']} Evaluation ---")
        print(f"Accuracy:  {res['Accuracy']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall:    {res['Recall']:.4f}")
        print(f"F1 Score:  {res['F1 Score']:.4f}")
    
    # 8. Learning Curve (Sequential as it's a different analysis)
    run_learning_curve(train_data, eval_data)
    
    # 6. Extended Lesk
    el_preds, el_true = run_lesk_evaluation(eval_data, method='extended')
    evaluate_model("Extended Lesk", el_preds, el_true)

if __name__ == "__main__":
    main()