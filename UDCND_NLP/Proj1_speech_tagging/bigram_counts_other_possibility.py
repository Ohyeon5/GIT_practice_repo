# other possibilities

def bigram_counts(sequences):
   counts = Counter()
   counts.update(chain(*(zip(s[:-1], s[1:]) for s in sequences)))
   return counts
tag_bigrams = bigram_counts(data.training_set.Y)