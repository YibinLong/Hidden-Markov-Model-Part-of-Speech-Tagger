# Hidden Markov Model Part of Speech Tagger
This Part-of-Speech tagger predicts the Part-of-Speech tags for untagged text by inference through a Hidden Markov Model.

The part-of-speech of a word or punctuation mark tells us its function in a sentence. For example, a noun is a person, place, or thing. A conjunction joins two clauses together, like "and" or "but".

The tagger processes training text files that have been tagged with the correct part-of-speech for each word. It then creates the initial, transition and observation probability tables in the Hidden Markov Model. After this, it uses the Viterbi algorithm to predict the part-of-speech tags for untagged text files. Given sufficiently large training and test sets (40,000+ characters), the tagger can achieve an accuracy of over 90%.

## Usage

1. Clone the repository and navigate to the directory
2. Use the following command to run the program, specifying the necessary input files and the output file:
`python3 tagger.py --trainingfiles <training files> --testfile <test file> --outputfile <output file>`
- Replace <training files> with one or more training text files separated by spaces.
- Replace <test file> with the test text file you want to evaluate.
- Replace <output file> with the name of the output text file where the program will write the predicted POS tags.

## Example
Let's say we have two training files named train1.txt and train2.txt (both labeled with POS tags), and we want to test our program on test.txt (not labeled with POS tags), generating an output file named output.txt. We would run the following command:
`python3 tagger.py --trainingfiles train1.txt train2.txt --testfile test.txt --outputfile output.txt`


