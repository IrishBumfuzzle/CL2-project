import csv
import os
import math
import random
from collections import defaultdict

def simplify_pos(pos):
    if not pos: return None
    if pos.startswith('N'): return 'n'
    if pos.startswith('V'): return 'v'
    if pos.startswith('J'): return 'a'
    if pos.startswith('R'): return 'r'
    return None

def load_data(data_dir):
    sentences = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                current_sent_id = None
                current_sent = []
                
                for row in reader:
                    # Unique sentence identifier: tagfile + pnum + snum
                    sent_id = (row['tagfile'], row['pnum'], row['snum'])
                    
                    if sent_id != current_sent_id:
                        if current_sent:
                            sentences.append(current_sent)
                        current_sent = []
                        current_sent_id = sent_id
                    
                    # Store relevant info
                    # We treat 'wnsn' as the sense label.
                    # Some entries might have None or empty strings.
                    sense = row['wnsn']
                    if sense == 'None' or sense == '':
                        sense = None
                    
                    token = {
                        'word': row['value'],
                        'lemma': row['lemma'],
                        'sense': sense,
                        'pos': row['pos']
                    }
                    current_sent.append(token)
                
                if current_sent:
                    sentences.append(current_sent)
    return sentences

class NaiveBayesWSD:
    def __init__(self, use_pos=True):
        self.use_pos = use_pos
        self.sense_counts = defaultdict(lambda: defaultdict(int))
        
        # context_counts[key][sense][word] = count of times 'word' appears in context of 'lemma' with 'sense'
        self.context_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # total_context_words[key][sense] = total count of context words for this sense (for denominator)
        self.total_context_words = defaultdict(lambda: defaultdict(int))
        
        # vocabulary[key] = set of all unique words seen in context of this lemma (for smoothing)
        self.vocabulary = defaultdict(set)

    def train(self, sentences):
        print(f"Training Naive Bayes model (use_pos={self.use_pos})...")
        for sent in sentences:
            # Pre-process context words: lowercase, maybe remove None
            context_words = [t['word'].lower() for t in sent if t['word']]
            
            for i, token in enumerate(sent):
                lemma = token['lemma']
                sense = token['sense']
                pos = simplify_pos(token['pos'])
                
                # We only train on words that have a lemma and a sense annotation
                if lemma and lemma != 'None' and sense:
                    if self.use_pos:
                        if not pos: continue
                        key = (lemma, pos)
                    else:
                        key = lemma
                        
                    # Update prior counts
                    self.sense_counts[key][sense] += 1
                    
                    current_context = context_words[:i] + context_words[i+1:]
                    
                    for ctx_word in current_context:
                        self.context_counts[key][sense][ctx_word] += 1
                        self.vocabulary[key].add(ctx_word)
                        self.total_context_words[key][sense] += 1
        print("Training complete.")

    def predict(self, sentence, target_index):
        target_token = sentence[target_index]
        lemma = target_token['lemma']
        pos = simplify_pos(target_token['pos'])
        
        if self.use_pos:
            if not pos: return None
            key = (lemma, pos)
        else:
            key = lemma
        
        # If we haven't seen this lemma before, we can't predict
        if key not in self.sense_counts:
            return None
            
        possible_senses = self.sense_counts[key].keys()
        
        best_sense = None
        max_log_prob = -float('inf')
        
        context_words = [t['word'].lower() for t in sentence if t['word']]
        current_context = context_words[:target_index] + context_words[target_index+1:]
        
        for sense in possible_senses:
            prior = self.sense_counts[key][sense]
            if prior == 0:
                continue
            log_prob = math.log(prior)
            
            # P(Context | Sense)
            vocab_size = len(self.vocabulary[key])
            total_w_s = self.total_context_words[key][sense]
            
            for ctx_word in current_context:
                # Laplace smoothing
                count_w_s = self.context_counts[key][sense].get(ctx_word, 0)
                
                prob_w_s = (count_w_s + 1) / (total_w_s + vocab_size + 1) # +1 to vocab size for unknown words potentially
                log_prob += math.log(prob_w_s)
            
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_sense = sense
                
        return best_sense

def evaluate(model, sentences):
    print("Evaluating model...")
    correct = 0
    total = 0
    
    for sent in sentences:
        for i, token in enumerate(sent):
            lemma = token['lemma']
            true_sense = token['sense']
            
            if lemma and lemma != 'None' and true_sense:
                predicted_sense = model.predict(sent, i)
                
                if predicted_sense is not None:
                    if predicted_sense == true_sense:
                        correct += 1
                    total += 1
    
    if total == 0:
        return 0.0
    
    return correct / total

if __name__ == "__main__":
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
    
    model = NaiveBayesWSD()
    model.train(train_data)
    
    accuracy = evaluate(model, test_data)
    print(f"Accuracy on test set: {accuracy:.4f}")
