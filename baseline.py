from collections import defaultdict
from nb import simplify_pos

class MostFrequentSenseWSD:
    def __init__(self, use_pos=True):
        self.use_pos = use_pos
        # most_frequent[key] = sense
        self.most_frequent = {}

    def train(self, sentences):
        # counts[key][sense] = count
        counts = defaultdict(lambda: defaultdict(int))
        
        print(f"Training MFS model (use_pos={self.use_pos})...")
        for sent in sentences:
            for token in sent:
                lemma = token['lemma']
                sense = token['sense']
                pos = simplify_pos(token['pos'])
                
                if lemma and lemma != 'None' and sense:
                    if self.use_pos:
                        if not pos: continue
                        key = (lemma, pos)
                    else:
                        key = lemma
                    
                    counts[key][sense] += 1
        
        # Determine most frequent sense for each key
        for key, sense_counts in counts.items():
            best_sense = max(sense_counts.items(), key=lambda item: item[1])[0]
            self.most_frequent[key] = best_sense
            
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
            
        return self.most_frequent.get(key, None)
