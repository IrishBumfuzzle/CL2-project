from nltk.corpus import wordnet

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

