import argparse
import collections
import logging
import os
import sys
import subprocess
import pickle
import re

from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


def apply_bpe(input_file, output_file, bpe_codes, dropout=0.0, seed=42):
    """
    Applies BPE or BPE-dropout to the input file and writes the result to the output file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to save the BPE-processed file.
        bpe_codes (str): Path to the BPE codes file.
        dropout (float): Dropout probability for BPE-dropout (0.0 for standard BPE).
        seed (int): Random seed for reproducibility.
    """
    cmd = [
        'subword-nmt', 'apply-bpe',
        '--dropout', str(dropout),
        '--input', input_file,
        '--output', output_file,
        '--codes', bpe_codes,
        '--seed', str(seed)
    ]

    try:
        subprocess.run(cmd, check=True)
        logging.info(f"BPE{'-dropout' if dropout > 0.0 else ''} applied to {input_file}, saved as {output_file}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying BPE{'-dropout' if dropout > 0.0 else ''} to {input_file}: {e}")
        raise


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description='Data Pre-processing with BPE and BPE-Dropout')
    parser.add_argument('--source-lang', required=True, type=str, help='Source language code (e.g., fr)')
    parser.add_argument('--target-lang', required=True, type=str, help='Target language code (e.g., en)')

    parser.add_argument('--train-prefix-src', required=True, type=str,
                        help='Prefix for source training data (e.g., train.fr.txt)')
    parser.add_argument('--train-prefix-tgt', required=True, type=str,
                        help='Prefix for target training data (e.g., train.en.txt)')
    parser.add_argument('--valid-prefix-src', required=True, type=str,
                        help='Prefix for source validation data (e.g., valid.fr.txt)')
    parser.add_argument('--valid-prefix-tgt', required=True, type=str,
                        help='Prefix for target validation data (e.g., valid.en.txt)')
    parser.add_argument('--test-prefix-src', type=str, default=None,
                        help='Prefix for source test data (e.g., test.fr.txt)')
    parser.add_argument('--test-prefix-tgt', type=str, default=None,
                        help='Prefix for target test data (e.g., test.en.txt)')

    parser.add_argument('--dest-dir', required=True, type=str, help='Destination directory for preprocessed data')

    parser.add_argument('--threshold-src', default=2, type=int,
                        help='Map words appearing less than this threshold to unknown (source)')
    parser.add_argument('--num-words-src', default=-1, type=int, help='Maximum number of source words to retain')
    parser.add_argument('--threshold-tgt', default=2, type=int,
                        help='Map words appearing less than this threshold to unknown (target)')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='Maximum number of target words to retain')

    parser.add_argument('--vocab-src', type=str, default=None, help='Path to existing source dictionary')
    parser.add_argument('--vocab-trg', type=str, default=None, help='Path to existing target dictionary')

    parser.add_argument('--quiet', action='store_true', help='Suppress logging output')

    parser.add_argument('--bpe_codes', required=True, type=str, help='Path to the BPE codes file')
    parser.add_argument('--bpe_dropout', type=float, default=0.0,
                        help='Dropout probability for BPE-dropout (source only)')

    return parser.parse_args()


def make_split_datasets(lang, dictionary, is_source, args):
    """
    Applies BPE/BPE-dropout and creates binary datasets for a given language.

    Args:
        lang (str): Language code (e.g., 'fr', 'en').
        dictionary (Dictionary): Source or target language dictionary.
        is_source (bool): Whether the language is the source language.
        args (Namespace): Parsed command-line arguments.
    """
    if is_source:
        train_prefix = args.train_prefix_src
        valid_prefix = args.valid_prefix_src
        test_prefix = args.test_prefix_src
    else:
        train_prefix = args.train_prefix_tgt
        valid_prefix = args.valid_prefix_tgt
        test_prefix = args.test_prefix_tgt

    if train_prefix:
        input_train = train_prefix  # e.g., train.fr.txt or train.en.txt

        output_train_bpe = f"{input_train}.bpe"
        dropout = args.bpe_dropout if is_source else 0.0
        apply_bpe(input_train, output_train_bpe, args.bpe_codes, dropout=dropout, seed=42)

        output_train_bin = os.path.join(args.dest_dir, f'train.{lang}')
        make_binary_dataset(
            input_file=output_train_bpe,
            output_file=output_train_bin,
            dictionary=dictionary
        )

    if valid_prefix:
        input_valid = valid_prefix  # e.g., valid.fr.txt or valid.en.txt

        output_valid_bpe = f"{input_valid}.bpe"
        dropout = 0.0
        apply_bpe(input_valid, output_valid_bpe, args.bpe_codes, dropout=dropout, seed=42)

        output_valid_bin = os.path.join(args.dest_dir, f'valid.{lang}')
        make_binary_dataset(
            input_file=output_valid_bpe,
            output_file=output_valid_bin,
            dictionary=dictionary
        )

    if test_prefix:
        input_test = test_prefix  # e.g., test.fr.txt or test.en.txt

        output_test_bpe = f"{input_test}.bpe"
        dropout = 0.0
        apply_bpe(input_test, output_test_bpe, args.bpe_codes, dropout=dropout, seed=42)

        output_test_bin = os.path.join(args.dest_dir, f'test.{lang}')
        make_binary_dataset(
            input_file=output_test_bpe,
            output_file=output_test_bin,
            dictionary=dictionary
        )


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)

    # Load or build source dictionary
    if args.vocab_src and os.path.exists(args.vocab_src):
        src_dict = Dictionary.load(args.vocab_src)
        if not args.quiet:
            logging.info(f'Loaded existing source dictionary from {args.vocab_src}')
    else:
        # Build and save the source dictionary from source training data
        src_dict = build_dictionary([args.train_prefix_src])
        src_dict.save(os.path.join(args.dest_dir, f'dict.{args.source_lang}'))
        if not args.quiet:
            logging.info(f'Saved source dictionary to {os.path.join(args.dest_dir, f"dict.{args.source_lang}")}')

    # Load or build target dictionary
    if args.vocab_trg and os.path.exists(args.vocab_trg):
        tgt_dict = Dictionary.load(args.vocab_trg)
        if not args.quiet:
            logging.info(f'Loaded existing target dictionary from {args.vocab_trg}')
    else:
        # Build and save the target dictionary from target training data
        tgt_dict = build_dictionary([args.train_prefix_tgt])
        tgt_dict.save(os.path.join(args.dest_dir, f'dict.{args.target_lang}'))
        if not args.quiet:
            logging.info(f'Saved target dictionary to {os.path.join(args.dest_dir, f"dict.{args.target_lang}")}')

    # Apply BPE-dropout and create binary datasets for source language
    make_split_datasets(lang=args.source_lang, dictionary=src_dict, is_source=True, args=args)

    # Apply BPE-dropout and create binary datasets for target language
    make_split_datasets(lang=args.target_lang, dictionary=tgt_dict, is_source=False, args=args)



if __name__ == '__main__':
    args = get_args()

    # Set up logging
    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("preprocess.log", mode='a')
            ]
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    main(args)
