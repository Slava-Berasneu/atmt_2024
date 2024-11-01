import os
import logging
import argparse
import pickle

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
import random
import re
from subword_nmt.apply_bpe import BPE

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='indomain/preprocessed_data/', help='path to data directory')
    parser.add_argument('--source-lang', default='fr', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')

    # Add model arguments
    parser.add_argument('--arch', default='lstm', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=10000, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints_asg4', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Add BPE arguments
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='Dropout probability for BPE-dropout')
    parser.add_argument('--bpe_codes', type=str, required=True, help='Path to the BPE codes file')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    """ Main training function with online BPE-dropout. """

    logging.info('Commencing training!')
    torch.manual_seed(42)

    utils.init_logging(args)

    # Load dictionaries
    src_dict_path = os.path.join(args.data, f'dict.{args.source_lang}')
    tgt_dict_path = os.path.join(args.data, f'dict.{args.target_lang}')

    src_dict = Dictionary.load(src_dict_path)
    logging.info(f'Loaded a source dictionary ({args.source_lang}) with {len(src_dict)} words')

    tgt_dict = Dictionary.load(tgt_dict_path)
    logging.info(f'Loaded a target dictionary ({args.target_lang}) with {len(tgt_dict)} words')

    def load_data(src_file, tgt_file):
        return Seq2SeqDataset(
            src_file=src_file,
            tgt_file=tgt_file,
            src_dict=src_dict,
            tgt_dict=tgt_dict
        )

    # Load validation dataset
    valid_src_pickle = os.path.join(args.data, f'valid.{args.source_lang}')
    valid_tgt_pickle = os.path.join(args.data, f'valid.{args.target_lang}')

    valid_dataset = load_data(
        src_file=valid_src_pickle,
        tgt_file=valid_tgt_pickle
    )

    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)
    logging.info(f'Built a model with {sum(p.numel() for p in model.parameters())} parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    # Path to the BPE codes file
    bpe_codes_file = args.bpe_codes

    for epoch in range(last_epoch + 1, args.max_epoch):
        logging.info(f"Starting epoch {epoch}")

        # Before each epoch, re-segment the training data with BPE-dropout
        src_file_original = os.path.join(args.data, f'train.{args.source_lang}')
        tgt_file_original = os.path.join(args.data, f'train.{args.target_lang}')

        prepared_dir = os.path.join(args.data, "prepared")
        os.makedirs(prepared_dir, exist_ok=True)

        src_file_bpe = os.path.join(prepared_dir, f'train.bpedropout.{args.source_lang}.epoch{epoch}')
        tgt_file_bpe = os.path.join(prepared_dir, f'train.bpedropout.{args.target_lang}.epoch{epoch}')

        logging.info("Applying BPE-dropout to training data...")
        resegment_data(
            src_file=src_file_original,
            tgt_file=tgt_file_original,
            output_src_file=src_file_bpe,
            output_tgt_file=tgt_file_bpe,
            dropout_prob=args.dropout_prob,
            bpe_codes_file=bpe_codes_file,
            src_dict=src_dict,  # Pass source dictionary
            tgt_dict=tgt_dict,  # Pass target dictionary
            seed=42 + epoch  # Different seed per epoch for variability
        )
        logging.info("BPE-dropout applied.")

        # Reload the training dataset with re-segmented data
        train_dataset = load_data(src_file=src_file_bpe, tgt_file=tgt_file_bpe)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=1,
            collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(
                train_dataset, args.max_tokens, args.batch_size, 1, 0, shuffle=False, seed=42
            )
        )

        model.train()
        stats = OrderedDict([
            ('loss', 0),
            ('lr', 0),
            ('num_tokens', 0),
            ('batch_size', 0),
            ('grad_norm', 0),
            ('clip', 0)
        ])

        # Display progress
        progress_bar = tqdm(train_loader, desc=f'| Epoch {epoch:03d}', leave=False, disable=False)

        # Iterate over the training set
        for i, sample in enumerate(progress_bar):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            model.train()

            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1)) / len(
                sample['src_lengths'])

            if torch.isnan(loss).any():
                logging.warning('Loss is NAN!')
                print(src_dict.string(sample['src_tokens'].tolist()[0]), '---',
                      tgt_dict.string(sample['tgt_tokens'].tolist()[0]))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Update statistics for progress bar
            total_loss = loss.item()
            num_tokens = sample['num_tokens']
            batch_size = len(sample['src_tokens'])

            stats['loss'] += total_loss * len(sample['src_lengths']) / sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(
            key + ' {:.4g}'.format(value / len(progress_bar)) for key, value in stats.items())
                                               ))

        # Calculate validation loss
        valid_perplexity = validate(args, model, criterion, valid_dataset, epoch)
        model.train()

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_perplexity)

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info(f'No validation set improvements observed for {args.patience} epochs. Early stop!')
            break


def resegment_data(src_file, tgt_file, output_src_file, output_tgt_file, dropout_prob, bpe_codes_file, src_dict, tgt_dict, seed=42):
    assert 0.0 <= dropout_prob < 1.0, "dropout_prob must be in the range [0.0, 1.0)."

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    random.seed(seed)

    # Initialize BPE with dropout
    try:
        with open(bpe_codes_file, 'r', encoding='utf-8') as bpe_codes:
            bpe = BPE(bpe_codes, vocab=None, glossaries=None)
        logging.info("BPE initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize BPE: {e}")
        raise

    def apply_bpe_dropout(line_array, dictionary):
        if not isinstance(line_array, np.ndarray):
            raise TypeError(f"Expected line_array to be a NumPy array, but got {type(line_array)}")

        sentence = dictionary.string(line_array.tolist())

        # Apply BPE with dropout
        bpe_line = bpe.process_line(sentence.strip())

        # Re-encode the processed string back to token IDs
        tokens = dictionary.binarize(bpe_line, tokenizer=lambda x: x.split(), append_eos=True)

        # Convert tokens to NumPy array
        processed_tokens = tokens.numpy()

        return processed_tokens

    # Process Source File
    logging.info(f"Processing source file: {src_file}")
    try:
        with open(src_file, 'rb') as src_in:
            src_data = pickle.load(src_in)
            if not isinstance(src_data, list):
                raise TypeError(f"Expected source pickle to contain a list, but got {type(src_data)}")
        logging.info(f"Loaded {len(src_data)} lines from source pickle.")
    except Exception as e:
        logging.error(f"Error loading source pickle file: {e}")
        raise

    src_processed = []
    for idx, line in enumerate(src_data):
        try:
            bpe_line = apply_bpe_dropout(line, src_dict)
            src_processed.append(bpe_line)
            if (idx + 1) % 10000 == 0:
                logging.info(f"Processed {idx + 1} source lines.")
        except Exception as e:
            logging.warning(f"Error processing source line {idx}: {e}")
            src_processed.append(np.array([src_dict.unk_idx], dtype=np.int32))

    temp_src_out = output_src_file + '.tmp'
    try:
        with open(temp_src_out, 'wb') as src_out:
            pickle.dump(src_processed, src_out, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(temp_src_out, output_src_file)
        logging.info(f"Saved processed source data to {output_src_file}.")
    except Exception as e:
        logging.error(f"Error saving processed source pickle file: {e}")
        if os.path.exists(temp_src_out):
            os.remove(temp_src_out)
        raise

    # Process Target File
    logging.info(f"Processing target file: {tgt_file}")
    try:
        with open(tgt_file, 'rb') as tgt_in:
            tgt_data = pickle.load(tgt_in)
            if not isinstance(tgt_data, list):
                raise TypeError(f"Expected target pickle to contain a list, but got {type(tgt_data)}")
        logging.info(f"Loaded {len(tgt_data)} lines from target pickle.")
    except Exception as e:
        logging.error(f"Error loading target pickle file: {e}")
        raise

    tgt_processed = []
    for idx, line in enumerate(tgt_data):
        try:
            bpe_line = apply_bpe_dropout(line, tgt_dict)
            tgt_processed.append(bpe_line)
            if (idx + 1) % 10000 == 0:
                logging.info(f"Processed {idx + 1} target lines.")
        except Exception as e:
            logging.warning(f"Error processing target line {idx}: {e}")
            tgt_processed.append(np.array([tgt_dict.unk_idx], dtype=np.int32))

    temp_tgt_out = output_tgt_file + '.tmp'
    try:
        with open(temp_tgt_out, 'wb') as tgt_out:
            pickle.dump(tgt_processed, tgt_out, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(temp_tgt_out, output_tgt_file)
        logging.info(f"Saved processed target data to {output_tgt_file}.")
    except Exception as e:
        logging.error(f"Error saving processed target pickle file: {e}")
        if os.path.exists(temp_tgt_out):
            os.remove(temp_tgt_out)
        raise

    logging.info("BPE-dropout re-segmentation completed successfully.")


def validate(args, model, criterion, valid_dataset, epoch):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=42))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    # Iterate over the validation set
    for i, sample in enumerate(valid_loader):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            # Compute loss
            output, attn_scores = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))

    return perplexity


if __name__ == '__main__':
    args = get_args()
    args.device_id = 0

    # Set up logging to file
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    main(args)
