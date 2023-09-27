# Hidden Markov Model Part of Speech Tagger
This Part-of-Speech tagger predicts the Part-of-Speech tags for untagged text by inference through a Hidden Markov Model.

The part-of-speech of a word or punctuation mark tells us its function in a sentence. For example, a noun is a person, place, or thing. A conjunction joins two clauses together, like "and" or "but".

The tagger processes training text files that have been tagged with the correct part-of-speech for each word. It then creates the initial, transition and observation probability tables in the Hidden Markov Model. After this, it uses the Viterbi algorithm to predict the part-of-speech tags for untagged text files. Given sufficiently large training and test sets (40,000+ characters), the tagger can achieve an accuracy of over 90%.

