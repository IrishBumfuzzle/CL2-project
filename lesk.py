from nltk.corpus import wordnet
import math
from collections import Counter, defaultdict
import numpy as np
import gensim.downloader as api

def simple_lesk(context_sentence, ambiguous_word, pos=None):
    context = set(context_sentence)
    synsets = wordnet.synsets(ambiguous_word, lang="eng")

    if pos:
        synsets = [ss for ss in synsets if ss.pos() == pos]

    if not synsets:
        return None

    sense = max(
        synsets, key=lambda ss: len(context.intersection(ss.definition().split()))
    )

    return sense

def extended_lesk(context_sentence, ambiguous_word, pos=None):
    context = set(context_sentence)
    synsets = wordnet.synsets(ambiguous_word, lang="eng")

    if pos:
        synsets = [ss for ss in synsets if ss.pos() == pos]

    if not synsets:
        return None

    def signature(ss):
        signature_words = set(ss.definition().split())
        for example in ss.examples():
            signature_words.update(example.split())
        for related_ss in ss.hypernyms() + ss.hyponyms() + ss.part_meronyms() + ss.part_holonyms():
            signature_words.update(related_ss.definition().split())
            for example in related_ss.examples():
                signature_words.update(example.split())
        return signature_words

    sense = max(
        synsets, key=lambda ss: len(context.intersection(signature(ss)))
    )

    return sense

class TfidfLesk:
    def __init__(self):
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0
        
    def fit(self, sentences):
        print("Computing IDF scores...")
        self.total_docs = len(sentences)
        for sent in sentences:
            # Treat each sentence as a document
            unique_words = set(t['word'].lower() for t in sent if t['word'])
            for word in unique_words:
                self.doc_freqs[word] += 1
        print("IDF computation complete.")
        
    def get_idf(self, word):
        # IDF = log(N / (df + 1))
        df = self.doc_freqs.get(word.lower(), 0)
        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def predict(self, context_sentence, ambiguous_word, pos=None):
        synsets = wordnet.synsets(ambiguous_word, lang="eng")
        if pos:
            synsets = [ss for ss in synsets if ss.pos() == pos]
        if not synsets:
            return None
            
        context_words = [w.lower() for w in context_sentence]
        
        best_sense = None
        max_score = -1
        
        for ss in synsets:
            # Signature includes definition and examples
            signature = ss.definition().split()
            for ex in ss.examples():
                signature.extend(ex.split())
            
            # Calculate overlap score weighted by IDF
            score = 0
            # We can use cosine similarity of TF-IDF vectors, or just sum of IDFs of overlapping words
            # Let's use sum of IDFs of overlapping words (Weighted Overlap)
            
            # Count frequencies in signature
            sig_counts = Counter([w.lower() for w in signature])
            
            # Count frequencies in context
            ctx_counts = Counter(context_words)
            
            # Intersection
            common_words = set(sig_counts.keys()) & set(ctx_counts.keys())
            
            for word in common_words:
                # TF in context * TF in signature * IDF^2 ? 
                # Or just sum(IDF) for each common token instance?
                # Standard "Lesk with TF-IDF" often means:
                # Score = sum_{w in overlap} (IDF(w))
                
                # Let's do: sum(IDF(w)) for each common word type
                score += self.get_idf(word)
            
            if score > max_score:
                max_score = score
                best_sense = ss
                
        return best_sense

class EmbeddingLesk:
    def __init__(self, model_name="glove-wiki-gigaword-50"):
        print(f"Loading word embeddings ({model_name})...")
        try:
            self.wv = api.load(model_name)
            print("Embeddings loaded.")
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            self.wv = None

    def get_vector(self, words):
        if not self.wv:
            return np.zeros(50)
            
        vectors = []
        for w in words:
            w = w.lower()
            if w in self.wv:
                vectors.append(self.wv[w])
        
        if not vectors:
            return np.zeros(self.wv.vector_size)
            
        return np.mean(vectors, axis=0)

    def predict(self, context_sentence, ambiguous_word, pos=None):
        if not self.wv:
            return None
            
        synsets = wordnet.synsets(ambiguous_word, lang="eng")
        if pos:
            synsets = [ss for ss in synsets if ss.pos() == pos]
        if not synsets:
            return None
            
        # Context vector
        # Exclude the ambiguous word itself from context?
        context_words = [w for w in context_sentence if w.lower() != ambiguous_word.lower()]
        context_vec = self.get_vector(context_words)
        
        best_sense = None
        max_sim = -2.0 # Cosine sim is between -1 and 1
        
        for ss in synsets:
            # Signature vector
            signature = ss.definition().split()
            for ex in ss.examples():
                signature.extend(ex.split())
                
            sig_vec = self.get_vector(signature)
            
            # Cosine similarity
            norm_c = np.linalg.norm(context_vec)
            norm_s = np.linalg.norm(sig_vec)
            
            if norm_c == 0 or norm_s == 0:
                sim = 0
            else:
                sim = np.dot(context_vec, sig_vec) / (norm_c * norm_s)
                
            if sim > max_sim:
                max_sim = sim
                best_sense = ss
                
        return best_sense


