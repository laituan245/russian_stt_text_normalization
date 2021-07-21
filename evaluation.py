import re
import string
import numpy as np

from tqdm import tqdm
from normalizer import Normalizer
from argparse import ArgumentParser

# Helper Function
def remove_punctuation(word: str):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('[' + all_punct_marks + ']', '', word)

def read_tsv(tsv_fp):
    print(f'Reading file from {tsv_fp}')

    # Extract instances
    instances = []
    with open(tsv_fp, 'r', encoding='utf8') as f:
        cur_classes, cur_tokens, cur_outputs = [], [], []
        for linectx, line in tqdm(enumerate(f)):
            es = line.strip().split('\t')
            if len(es) == 2 and es[0] == '<eos>':
                instances.append((cur_classes, cur_tokens, cur_outputs))
                # Reset
                cur_classes, cur_tokens, cur_outputs = [], [], []
                continue
            # Update the current example
            assert len(es) == 3
            cur_classes.append(es[0])
            cur_tokens.append(es[1])
            cur_outputs.append(es[2])
    print(f'Number of instances: {len(instances)}')

    # Extract written_strs and spoken_strs
    written_strs, spoken_strs = [], []
    for inst in instances:
        cur_classes, cur_tokens, cur_outputs = inst
        # written_str
        filtered_cur_tokens = []
        for t, o in zip(cur_tokens, cur_outputs):
            if o == 'sil': continue
            filtered_cur_tokens.append(t)
        written_str = ' '.join(filtered_cur_tokens)

        # spoken_str
        filtered_cur_outputs = []
        for t, o in zip(cur_tokens, cur_outputs):
            if o in ['<self>']: filtered_cur_outputs.append(t)
            elif o == 'sil': continue
            else: filtered_cur_outputs = filtered_cur_outputs + o.split(' ')
        spoken_str = ' '.join(filtered_cur_outputs)
        # Update written_strs and spoken_strs
        written_strs.append(written_str)
        spoken_strs.append(spoken_str)
    assert(len(written_strs) == len(instances))
    assert(len(spoken_strs) == len(instances))

    return written_strs, spoken_strs


# Main code
if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument('--test_fp', type=str,
                        default='resources/test.tsv',
                        help='Path to the test data file')
    args = parser.parse_args()

    # Read
    written_strs, spoken_strs = read_tsv(args.test_fp)

    # Evaluation
    error_file = open('errors.txt', 'w+', encoding='utf-8')
    total_count, correct_count = 0, 0
    norm = Normalizer()
    for i in tqdm(range(len(written_strs))):
        w_str, s_str = written_strs[i], spoken_strs[i]
        pred_str = norm.norm_text(w_str)
        normalized_pred_str = remove_punctuation(pred_str.strip().replace(' ', ''))
        normalized_target_str = remove_punctuation(s_str.strip().replace(' ', ''))
        if normalized_pred_str == normalized_target_str:
            correct_count += 1
        else:
            error_file.write('Written Input: {}\n'.format(w_str))
            error_file.write('Spoken Target: {}\n'.format(s_str))
            error_file.write('Predicted Spoken: {}\n'.format(pred_str))
            error_file.write('\n')
        total_count += 1
    print(f'total_count: {total_count}')
    print(f'correct_count: {correct_count}')
    acc = correct_count / total_count
    print(f'Accuracy: {acc}')
    error_file.close()
