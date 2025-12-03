import csv
import os
import random
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def simplify_pos(pos):
    if not pos: return None
    if pos.startswith('N'): return 'n'
    if pos.startswith('V'): return 'v'
    if pos.startswith('J'): return 'a'
    if pos.startswith('R'): return 'r'
    return None

def load_data(data_dir):
    sentences = []
    # Iterate over all files in data_dir
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

class SVMWSD:
    def __init__(self, use_pos=True):
        self.use_pos = use_pos
        self.models = {}

    def train(self, sentences):
        # Organize data by key
        lemma_data = defaultdict(list) # key -> list of (context_dict, sense)
        
        print(f"Extracting features (use_pos={self.use_pos})...")
        for sent in sentences:
            context_words = [t['word'].lower() for t in sent if t['word']]
            
            for i, token in enumerate(sent):
                lemma = token['lemma']
                sense = token['sense']
                pos = simplify_pos(token['pos'])
                
                if lemma and lemma != 'None' and sense:
                    if self.use_pos:
                        if not pos: continue
                        key = (lemma, pos)
                    else:
                        key = lemma
                        
                    # Feature extraction
                    features = {}
                    current_context = context_words[:i] + context_words[i+1:]
                    for word in current_context:
                        features[word] = features.get(word, 0) + 1
                    
                    lemma_data[key].append((features, sense))
        
        print(f"Training SVM models for {len(lemma_data)} keys...")
        
        for key, examples in lemma_data.items():
            X_dicts = [x for x, y in examples]
            y_labels = [y for x, y in examples]
            
            unique_labels = set(y_labels)
            if len(unique_labels) == 1:
                # Trivial case: only one sense seen
                self.models[key] = list(unique_labels)[0]
                continue
                
            # Create pipeline
            # DictVectorizer converts dicts of features to vectors
            # LinearSVC is the SVM
            pipeline = Pipeline([
                ('vect', DictVectorizer()),
                ('clf', LinearSVC(random_state=42, dual='auto', max_iter=100000))
            ])
            
            pipeline.fit(X_dicts, y_labels)
            self.models[key] = pipeline
            
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
        
        if key not in self.models:
            return None
            
        model = self.models[key]
        
        # If model is just a string, it means we only saw one sense
        if isinstance(model, str):
            return model
            
        # Extract features
        context_words = [t['word'].lower() for t in sentence if t['word']]
        current_context = context_words[:target_index] + context_words[target_index+1:]
        features = {}
        for word in current_context:
            features[word] = features.get(word, 0) + 1
            
        return model.predict([features])[0]

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
    
    model = SVMWSD()
    model.train(train_data)
    
    accuracy = evaluate(model, test_data)
    print(f"Accuracy on test set: {accuracy:.4f}")
