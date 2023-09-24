import os
import sys
import argparse
import numpy as np
import time

np.set_printoptions(threshold=sys.maxsize)

def parse_line(line):
  line = line.split()  # turn line into ['WORD', ':', 'POS']
  return line

def viterbi(E, S, I, T, M):
  '''
  Based off of the viterbi pseudocode
  
  E = set (LIST) of observations over time steps (words)
  S = set (LIST) of hidden state values (tags)
  I = initial probabilities
  T = transition matrix
  M = observation matrix 
  '''
  prob = np.zeros((len(E), len(S)))  # all the words in a sentence by 91 tags
  prev = np.zeros((len(E), len(S)))

  smooth = 1e-20

  # Determine values for time step 0

  for i, tag in enumerate(S):
    # print(M[tag].get(E[0]), i, tag)
    
    if M[tag].get(E[0]) != None:
      prob[0, i] = I[i] * M[tag][E[0]]  # E[0] is the first word
      prev[0, i] = None

    else:

      # Laplace Smoothing:
      prob[0, i] = I[i] * smooth
      prev[0, i] = None

  for t in range(1, len(E)):
    for i, tag_i in enumerate(S):

      if M[tag_i].get(E[t]) != None:
        x = np.argmax(prob[t - 1, :] * T[:, i] * M[tag_i][E[t]])
        prob[t, i] = prob[t - 1, x] * T[x, i] * M[tag_i][E[t]]
        prev[t, i] = x
      else:

        # Laplace Smoothing:
        x = np.argmax(prob[t - 1, :] * T[:, i] * smooth)
        prob[t, i] = prob[t - 1, x] * T[x, i] * smooth
        prev[t, i] = x
        
  return prob, prev


tags = [
  "AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
  "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
  "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
  "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
  'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
  'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
  'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI',
  'NN1-NP0', 'NN1-VVB', 'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0',
  'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP', 'CJS-AVQ', 'PRP-CJS', 'DT0-CJT',
  'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD'
]

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--trainingfiles",
                      action="append",
                      nargs="+",
                      required=True,
                      help="The training files.")
  parser.add_argument("--testfile",
                      type=str,
                      required=True,
                      help="One test file.")
  parser.add_argument("--outputfile",
                      type=str,
                      required=True,
                      help="The output file.")
  args = parser.parse_args()

  training_list = args.trainingfiles[0]

  start_create_matrices_time = time.time()

  # Parse the first line
  # The initial probabilities over the POS tags (how likely each POS tag appears at the beginning of a sentence)
  initial_probs_count = np.zeros(91)
  for training_file in training_list:
    read_file = open(training_file, "r")

    first_line = parse_line(read_file.readline())

    tag_idx = tags.index(first_line[2])

    initial_probs_count[tag_idx] += 1

    read_file.close()


  # Calculate the rest of INITIAL_PROB_COUNT
  next_is_first_word = float('-inf')
  for training_file in training_list:
    read_file = open(training_file, "r")
    lines = read_file.readlines()

    ctr = 0
    for line in lines:
      line = parse_line(line)

      if line[0] in ['.', '?', '!']:
        next_is_first_word = ctr

      if ctr == next_is_first_word + 1:
        tag_idx = tags.index(line[2])

        initial_probs_count[tag_idx] += 1

      ctr += 1

    read_file.close()

  # Calculate the probabilities of INITIAL_PROB
  initial_probs = np.zeros(91)
  total_num_of_initial = 0
  for val in initial_probs_count:
    total_num_of_initial += val

  for i in range(91):
    initial_probs[i] = initial_probs_count[i] / total_num_of_initial

  # For the transition probability table, for each starting tag, there is a probability distribution over the next tag.
  transition_matrix_count = np.zeros((91, 91))

  # NOTE: Let row = the current tag, col = the chance of the NEXT tag

  for training_file in training_list:
    read_file = open(training_file, "r")
    lines = read_file.readlines()

    prev_tag = parse_line(lines[0])[2]

    for line in lines[1:]:
      line = parse_line(line)

      row_index = tags.index(prev_tag)
      col_index = tags.index(line[2])

      transition_matrix_count[row_index, col_index] += 1

      prev_tag = line[2]

    read_file.close()


  # Calculate the probabilities of TRANSITION_PROB

  transition_probs = np.zeros((91, 91))

  row_sum_lst = []
  for row in range(91):
    row_sum = transition_matrix_count[row].sum()
    row_sum_lst.append(row_sum)

  for row in range(91):
    for col in range(91):
      if row_sum_lst[row] != 0:
        transition_probs[row, col] = transition_matrix_count[row, col] / row_sum_lst[row]
      else:
        # LAPLACE SMOOTHING
        # assign a small value to each transition that never occurs
        transition_probs[row, col] = 1e-20

  obs_prob_count_dict = {}

  for tag in tags:
    obs_prob_count_dict[tag] = {}

  for training_file in training_list:
    read_file = open(training_file, "r")
    lines = read_file.readlines()

    for line in lines:
      line = parse_line(line)

      cur_tag = line[2]
      cur_word = line[0]

      if cur_word in obs_prob_count_dict[cur_tag]:
        obs_prob_count_dict[cur_tag][cur_word] += 1
      else:
        obs_prob_count_dict[cur_tag][cur_word] = 1

    read_file.close()

  # Calculate the probabilities of OBSERVATION_PROB
  sum_values = 0
  for tag in obs_prob_count_dict:
    # sum up the values
    sum_values = sum(obs_prob_count_dict[tag].values())

    for word in obs_prob_count_dict[tag]:
      obs_prob_count_dict[tag][word] /= sum_values

  # NOTE: the obs_prob_count_dict was changed in-place, so the name is no longer accurate
  obs_prob_dict = obs_prob_count_dict

  end_create_matrices_time = time.time()
  print("Time to create initial matrices: ", end_create_matrices_time - start_create_matrices_time)

  # FINAL VITERBI LOOP
  start_time = time.time()

  output_file = open(args.outputfile, "w")

  read_test = open(args.testfile, "r")
  lines = read_test.readlines()

  # Make E[], E = set (LIST) of observations over time steps (words) 
  E = []
  for line in lines:
    line = line.strip()
    E.append(line)

    if line in ['.', '?', '!']:

      # viterbi loop
      prob, prev = viterbi(E, tags, initial_probs, transition_probs, obs_prob_dict)

      reversed_tag_lst = []

      max_prob_idx = np.argmax(prob[len(E) - 1, :])

      reversed_tag_lst.append(tags[max_prob_idx])

      for t in reversed(range(1, len(E))):
        max_prob_idx = int(prev[t, max_prob_idx])
        reversed_tag_lst.append(tags[max_prob_idx])

      for t, word in enumerate(E):
        output_file.write(word + " : " + reversed_tag_lst[len(E) - 1 - t] + "\n")

      E = []  # reset E at the end

  output_file.close()

  end_time = time.time()

  print("Time to run Viterbi: ", end_time - start_time)
