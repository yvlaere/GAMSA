import itertools
import random
import numpy as np

def generate_G0(sta, n):
    """
    Randomly creates a starting generation for the genetic algorithm
    :param sta: sequences to align
    :param n: population size
    :return G0: generation zero
    """
    # initialize
    G0 = []

    # repeat n times
    for i in range(n):
        G0.append({})
        # generate random offset (needed to create a difference between different alignments in the population
        offset = random.randint(0, 5)
        max_len = max([len(seq) for seq in sta.values()])
        # add gaps at the beginning of the sequence to make all sequences the same length
        for seq_name in sta:
            G0[i][seq_name] = '-'*(offset + max_len - len(sta[seq_name])) + sta[seq_name]

    return G0

def pairwise_score(seq1, seq2):
    """
    calculate the score of a pairwise alignment
    :param seq1: first sequence of the alignment
    :param seq2: second sequence of the alignment
    :return score: score of the pairwise alignment
    """

    gap_opening_cost = 5
    gap_extension_cost = 2
    mismatch_cost = 3
    match_cost = -1

    # initialize
    score = 0
    # iterate over all bases
    gap_1 = False
    gap_2 = False
    for k in range(len(seq1)):
        c1 = seq1[k]
        c2 = seq2[k]
        # mismatch cost
        if c1 != c2:
            score -= mismatch_cost
        else:
            score -= match_cost

        # gap cost for sequence 1
        if c1 == '-':
            # new gap
            if gap_1 == False:
                gap_1 = True
                score -= gap_opening_cost
            # gap extension
            else:
                score -= gap_extension_cost
        # reset gap
        elif c1 != '-':
            gap_1 = False

        # gap cost for sequence 2
        if c2 == '-':
            # new gap
            if gap_2 == False:
                gap_2 = True
                score -= gap_opening_cost
            # gap extension
            else:
                score -= gap_extension_cost
        # reset gap
        elif c2 != '-':
            gap_2 = False

    return score

def OF(ind):
    """
    Score an individual using an objective function
    :param ind: individual alignment
    :return score: score for the alignment
    """
    # initialize
    score = 0

    # get every pairwise combination of sequences
    for i in range(1, len(ind)):
        for j in range(i):
            # calculate pairwise score
            score += pairwise_score(ind[i], ind[j])

    return score

def one_point_crossover(ind1, ind2):
    """
    create a single point crossover between two individuals
    :param ind1: first individual alignment
    :param ind2: second individual alignment
    :return child: resulting individual alignment
    """

    # get length of shortest sequence without considering gaps(same for both individuals)
    min_len = min([len(seq.replace('-', '')) for seq in ind1.values()])
    max_len1 = max([len(seq) for seq in ind1.values()])
    max_len2 = max([len(seq) for seq in ind2.values()])
    max_len = max(max_len1, max_len2)

    # generate a random position for a crossover
    crossover_position = random.randint(0, min_len)

    # initialize
    ind1_begin = [""]*len(ind1.values())
    ind1_end = [""]*len(ind1.values())
    mod_ind1 = {key: "" for key in ind1.keys()}
    ind2_begin = [""]*len(ind2.values())
    ind2_end = [""]*len(ind2.values())
    mod_ind2 = {key: "" for key in ind2.keys()}

    # iterate over all sequences of individual 1
    for i in range(len(ind1.values())):
        charcount1 = 0
        position1 = 0
        # iterate over all characters in the sequence until the crossover position is reached
        while charcount1 < crossover_position:
            ind1_begin[i] += list(ind1.values())[i][position1]

            if list(ind1.values())[i][position1] != "-":
                charcount1 += 1

            position1 += 1

        # get the remainder of the sequences
        ind1_end[i] = list(ind1.values())[i][position1:]

    # iterate over all sequences of individual 2
    for i in range(len(ind2.values())):
        charcount2 = 0
        position2 = 0
        # iterate over all characters in the sequence until the crossover position is reached
        while charcount2 < crossover_position:
            ind2_begin[i] += list(ind2.values())[i][position2]

            if list(ind2.values())[i][position2] != "-":
                charcount2 += 1

            position2 += 1

        # get the remainder of the sequences
        ind2_end[i] = list(ind2.values())[i][position2:]

    # check if all sequences within a part have the same length
    ind1_begin_lens = [len(seq) for seq in ind1_begin]
    ind1_end_lens = [len(seq) for seq in ind1_end]
    ind2_begin_lens = [len(seq) for seq in ind2_begin]
    ind2_end_lens = [len(seq) for seq in ind2_end]

    for i in range(len(ind1_begin)):
        gap_size = max(ind1_begin_lens) - len(ind1_begin[i])
        ind1_begin[i] = ind1_begin[i] + '-'*gap_size

    for i in range(len(ind2_begin)):
        gap_size = max(ind2_begin_lens) - len(ind2_begin[i])
        ind2_begin[i] = ind2_begin[i] + '-'*gap_size

    for i in range(len(ind1_end)):
        gap_size = max(ind1_end_lens) - len(ind1_end[i])
        ind1_end[i] = '-'*gap_size + ind1_end[i]

    for i in range(len(ind2_end)):
        gap_size = max(ind2_end_lens) - len(ind2_end[i])
        ind2_end[i] = '-' * gap_size + ind2_end[i]

    # glue the sequences together
    # iterate over all sequences
    for i in range(len(ind1_begin)):
        mod_ind1[list(mod_ind1.keys())[i]] = ind1_begin[i] + ind2_end[i]
        mod_ind2[list(mod_ind2.keys())[i]] = ind2_begin[i] + ind1_end[i]

    # score both potential children and determine who is best
    if OF(list(mod_ind1.values())) < OF(list(mod_ind2.values())):
        child = mod_ind2
    else:
        child = mod_ind1

    return child

def uniform_crossover(ind1, ind2): # currently, blocks of ind1 and ind2 don't necessarily match, do we retry or give up when that happens???
    """
    create a uniform crossover between two individuals
    :param ind1: first individual alignment
    :param ind2: second individual alignment
    :return child: resulting individual alignment
    """

    # find consistent positions in individual 1
    consistent_pos1 = []
    for i in range(len(list(ind1.values())[0])):
        ref_char = list(ind1.values())[0][i]
        consistent = False
        if ref_char != "-":
            consistent = True
            # iterate over all sequences
            for j in range(len(ind1.keys())):
                if list(ind1.values())[j][i] != ref_char:
                    consistent = False

        if consistent:
            consistent_pos1.append(i)

    # find consistent positions in individual 2
    consistent_pos2 = []
    for i in range(len(list(ind2.values())[0])):
        ref_char = list(ind2.values())[0][i]
        consistent = False
        if ref_char != "-":
            consistent = True
            # iterate over all sequences
            for j in range(len(ind2.keys())):
                if list(ind2.values())[j][i] != ref_char:
                    consistent = False

        if consistent:
            consistent_pos2.append(i)

    # randomly choose a block for crossover
    # no crossover possible if there are less than 2 positions where the crossover could happen
    print('posses')
    print(len(consistent_pos1))
    print(len(consistent_pos2))
    crossover_possible = True
    if (len(consistent_pos1) <= 2) | (len(consistent_pos2) <= 2):
        crossover_possible = False
    else:
        if len(consistent_pos1) < len(consistent_pos2):
            crossover_block = random.randint(0, (len(consistent_pos1) - 2))
        else:
            crossover_block = random.randint(0, (len(consistent_pos2) - 2))

    if crossover_possible:
        # split children in sections
        # initialize
        ind1_begin = [""]*len(ind1.values())
        ind1_block = [""]*len(ind1.values())
        ind1_end = [""]*len(ind1.values())
        mod_ind1 = {key: "" for key in ind1.keys()}

        ind2_begin = [""]*len(ind2.values())
        ind2_block = [""]*len(ind2.values())
        ind2_end = [""]*len(ind2.values())
        mod_ind2 = {key: "" for key in ind2.keys()}

        # iterate over all sequences
        for i in range(len(ind1.keys())):
            ind1_begin[i] = list(ind1.values())[i][0:consistent_pos1[crossover_block]]
            ind1_block[i] = list(ind1.values())[i][consistent_pos1[crossover_block]:consistent_pos1[crossover_block + 1]]
            ind1_end[i] = list(ind1.values())[i][consistent_pos1[crossover_block + 1]:]

            ind2_begin[i] = list(ind2.values())[i][0:consistent_pos2[crossover_block]]
            ind2_block[i] = list(ind2.values())[i][consistent_pos2[crossover_block]:consistent_pos1[crossover_block + 1]]
            ind2_end[i] = list(ind2.values())[i][consistent_pos2[crossover_block + 1]:]

        # check if the block has the same content in both individuals, if it doesn't, don't do the crossover
        same_block = True
        for i in range(len(ind1_block)):
            if ind1_block[i].replace('-', '') != ind2_block[i].replace('-', ''):
                same_block = False

        if same_block:
            # glue the children together
            # iterate over all sequences
            for i in range(len(ind1_begin)):
                mod_ind1[list(mod_ind1.keys())[i]] = ind1_begin[i] + ind2_block[i] + ind1_end[i]
                mod_ind2[list(mod_ind2.keys())[i]] = ind2_begin[i] + ind1_block[i] + ind2_end[i]

            # score both potential children and determine who is best
            if OF(list(mod_ind1.values())) < OF(list(mod_ind2.values())):
                child = mod_ind2
            else:
                child = mod_ind1
        else:
            # return first parent
            child = ind1
    else:
        # return first parent
        child = ind1

    return child

def gap_insertion(ind, max_dist, max_gap_len):
    """
    insert gap in alignment
    :param ind: individual alignment
    :param max_dist: maximum distance between the gap insertions
    :param max_gap_len: maximum size of the gap
    :return mod_ind: modified individual alignment
    """

    # find two groups of sequences
    ind_vals = list(ind.values())
    G1_size = round(len(ind_vals)/2)
    indexes = list(range(len(ind_vals)))
    best_subset_score = float('-inf')
    for subset in itertools.combinations(indexes, G1_size):
        # calculate score of the subset
        if OF([ind_vals[ind] for ind in subset]) > best_subset_score:
            best_subset_score = OF([ind_vals[ind] for ind in subset])
            best_subset = subset

    # get P1, P2, gap_length
    P1 = random.randint(0, len(list(ind.values())[0]))
    P_dist = random.randint(-max_dist, max_dist)
    if P1 + P_dist < 0:
        P2 = 0
    elif P1 + P_dist > len(list(ind.values())[0]):
        P2 = len(list(ind.values())[0])
    else:
        P2 = P1 + P_dist
    gap_len = random.randint(1, max_gap_len)

    # introduce gaps
    child = {}

    for j in range(len(ind_vals)):
        # at P1
        if j in best_subset:
            child[list(ind.keys())[j]] = ind_vals[j][:P1] + '-'*gap_len + ind_vals[j][P1:]
        else:
            child[list(ind.keys())[j]] = ind_vals[j][:P2] + '-'*gap_len + ind_vals[j][P2:]

    return child

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parameters
    to_replace = 50 # percentage of parents that should be replaced each generation
    population_size = 100 # size of a population
    max_dist = 10
    max_gap_len = 2
    iterations = 200

    # read in sequences that should be aligned
    sta = {}
    with open('test_sequences.fasta', 'r') as f:
        line_counter = 0
        for line in f:
            if line_counter == 0:
                sequence_name = line.strip()
                line_counter = 1
            elif line_counter == 1:
                sequence = line.strip()
                sta[sequence_name] = sequence
                line_counter = 0

    # generate a generation zero
    G0 = generate_G0(sta, population_size)
    Gn = G0

    for k in range(iterations):
        # evaluate Gn (higher score means better alignment)
        scores = []
        for i in range(len(Gn)):
            scores.append(-1*OF(list(Gn[i].values())))

        # Create next generation
        # sort the current generation
        score_set = list(set(scores))
        score_set.sort()
        Gn_sorted = []
        for i in score_set:
            for j in range(len(scores)):
                if scores[j] == i:
                    Gn_sorted.append(Gn[j])

        # keep a fraction from previous generation containing the best alignments
        to_keep = round(population_size*(1 - to_replace/100))
        new_children = population_size - to_keep
        Gnext = Gn_sorted[0:to_keep]

        # score is used as a probability for each individual to be a parent
        min_score = min(scores)
        new_scores = [x + min_score for x in scores]
        sum_scores = sum(new_scores)
        percents = [x/sum_scores for x in new_scores]

        # choose an operator
        for i in range(new_children):
            random_operator = random.randint(0, 3)
            parent1, parent2 = np.random.choice(Gn_sorted, size = 2, p = percents)
            if random_operator == 0:
                print("one point crossover")
                Gnext.append(one_point_crossover(parent1, parent2))
                #Gnext.append(gap_insertion(parent1, max_dist, max_gap_len))
            elif random_operator == 1:
                print("uniform crossover")
                Gnext.append(uniform_crossover(parent1, parent2))
                #Gnext.append(gap_insertion(parent1, max_dist, max_gap_len))
            elif random_operator == 2:
                print("gap insertion")
                Gnext.append(gap_insertion(parent1, max_dist, max_gap_len))

        # update generation
        Gn = Gnext
        print('new generation')








    print('GO')
    print(G0)
    print('solution')

    # evaluate Gn (higher score means better alignment)
    scores = []
    for i in range(len(Gn)):
        scores.append(-1*OF(list(Gn[i].values())))

    # Create next generation
    # sort the current generation
    score_set = list(set(scores))
    score_set.sort()
    Gn_sorted = []
    for i in score_set:
        for j in range(len(scores)):
            if scores[j] == i:
                Gn_sorted.append(Gn[j])
    print(Gn_sorted[0])
    print('score')
    print(score_set[0])