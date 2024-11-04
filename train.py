import argparse
import collections
import logging
import os
import sys
import subprocess
import pickle
import gc

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch
from torch import nn
from collections import OrderedDict
from tqdm import tqdm


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
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='Dropout probability for BPE-dropout')
    parser.add_argument('--bpe_codes', type=str, required=True, help='Path to the BPE codes file')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def convert_pickled_to_plaintext(pickled_file, dictionary, plaintext_file):
    """
    Converts a pickled file containing token IDs to a plaintext file.

    Args:
        pickled_file (str): Path to the pickled file.
        dictionary (Dictionary): The dictionary to convert IDs to words.
        plaintext_file (str): Path to save the plaintext output.
    """
    with open(pickled_file, 'rb') as f:
        try:
            token_ids_list = pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load pickled file {pickled_file}: {e}")
            sys.exit(1)

    with open(plaintext_file, 'w', encoding='utf-8') as f_out:
        for token_ids in token_ids_list:
            words = dictionary.string(token_ids.tolist())
            f_out.write(words + '\n')


def validate(args, model, criterion, valid_dataset, epoch):
    """
    Validates the model on the validation dataset.

    Args:
        args: Command-line arguments.
        model: The model to validate.
        criterion: Loss criterion.
        valid_dataset: The validation dataset.
        epoch (int): Current epoch number.

    Returns:
        float: Validation perplexity.
    """
    logging.info(f"Validating at epoch {epoch}...")
    model.eval()
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collater,
        num_workers=1
    )

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in tqdm(valid_loader, desc='Validating', leave=False):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
            total_loss += loss.item()
            total_tokens += sample['num_tokens']

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    logging.info(f"Epoch {epoch}: Validation Perplexity: {perplexity:.4f}")
    return perplexity


def load_data(src_file, tgt_file, src_dict, tgt_dict):
    """Loads a Seq2SeqDataset from source and target binary files."""
    return Seq2SeqDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_dict=src_dict,
        tgt_dict=tgt_dict
    )

def preprocess_data(args, epoch_dir, fixed_train_src, fixed_valid_src, fixed_test_src,
                   fixed_train_tgt, fixed_valid_tgt, fixed_test_tgt,
                   src_dict_path_existing, tgt_dict_path_existing):
    """
    Preprocesses both source and target data by calling preprocess.py once with all required arguments.
    """
    preprocess_cmd = [
        'python', 'preprocess.py',
        '--source-lang', args.source_lang,
        '--target-lang', args.target_lang,
        '--train-prefix-src', fixed_train_src,  # e.g., train.fr.txt
        '--train-prefix-tgt', fixed_train_tgt,  # e.g., train.en.txt
        '--valid-prefix-src', fixed_valid_src,  # e.g., valid.fr.txt
        '--valid-prefix-tgt', fixed_valid_tgt,  # e.g., valid.en.txt
        '--test-prefix-src', fixed_test_src,    # e.g., test.fr.txt
        '--test-prefix-tgt', fixed_test_tgt,    # e.g., test.en.txt
        '--dest-dir', epoch_dir,                # Save to epoch-specific directory
        '--bpe_codes', args.bpe_codes,
        '--bpe_dropout', str(args.dropout_prob),  # e.g., 0.2 for source
        '--vocab-src', src_dict_path_existing,
        '--vocab-trg', tgt_dict_path_existing,
        '--quiet',
    ]

    logging.info("Running preprocessing command:")
    logging.info(" ".join(preprocess_cmd))

    try:
        subprocess.run(preprocess_cmd, check=True)
        logging.info("Preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)

def main(args):
    """ Main training function with on-the-fly BPE-dropout. """

    logging.info('Commencing training!')
    torch.manual_seed(42)

    utils.init_logging(args)

    src_dict_path = os.path.join(args.data, f'dict.{args.source_lang}')
    tgt_dict_path = os.path.join(args.data, f'dict.{args.target_lang}')

    src_dict = Dictionary.load(src_dict_path)
    logging.info(f'Loaded a source dictionary ({args.source_lang}) with {len(src_dict)} words')

    tgt_dict = Dictionary.load(tgt_dict_path)
    logging.info(f'Loaded a target dictionary ({args.target_lang}) with {len(tgt_dict)} words')

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

    # Define directories
    base_dir = args.data  # e.g., data/en-fr/prepared/
    prepared_dir = os.path.join(base_dir, "prepared")  # e.g., data/en-fr/prepared/prepared/
    os.makedirs(prepared_dir, exist_ok=True)

    # Define fixed plaintext file paths
    fixed_train_src = os.path.join(base_dir, f'train.{args.source_lang}.txt')
    fixed_train_tgt = os.path.join(base_dir, f'train.{args.target_lang}.txt')
    fixed_valid_src = os.path.join(base_dir, f'valid.{args.source_lang}.txt')
    fixed_valid_tgt = os.path.join(base_dir, f'valid.{args.target_lang}.txt')
    fixed_test_src = os.path.join(base_dir, f'test.{args.source_lang}.txt')
    fixed_test_tgt = os.path.join(base_dir, f'test.{args.target_lang}.txt')

    for epoch in range(last_epoch + 1, args.max_epoch):
        logging.info(f"Starting epoch {epoch}")

        epoch_dir = os.path.join(prepared_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Define paths to existing dictionaries
        src_dict_path_existing = src_dict_path  # dict.fr
        tgt_dict_path_existing = tgt_dict_path  # dict.en

        # Execute Preprocessing Commands
        preprocess_data(
            args=args,
            epoch_dir=epoch_dir,
            fixed_train_src=fixed_train_src,  # e.g., 'data/en-fr/prepared/train.fr.txt'
            fixed_valid_src=fixed_valid_src,  # e.g., 'data/en-fr/prepared/valid.fr.txt'
            fixed_test_src=fixed_test_src,  # e.g., 'data/en-fr/prepared/test.fr.txt'
            fixed_train_tgt=fixed_train_tgt,  # e.g., 'data/en-fr/prepared/train.en.txt'
            fixed_valid_tgt=fixed_valid_tgt,  # e.g., 'data/en-fr/prepared/valid.en.txt'
            fixed_test_tgt=fixed_test_tgt,  # e.g., 'data/en-fr/prepared/test.en.txt'
            src_dict_path_existing=src_dict_path_existing,  # Path to existing source dictionary
            tgt_dict_path_existing=tgt_dict_path_existing  # Path to existing target dictionary
        )

        # Define paths to the new preprocessed binary files after preprocessing
        src_file_bpe = os.path.join(epoch_dir, f'train.{args.source_lang}')
        tgt_file_bpe = os.path.join(epoch_dir, f'train.{args.target_lang}')
        valid_src_bpe = os.path.join(epoch_dir, f'valid.{args.source_lang}')
        valid_tgt_bpe = os.path.join(epoch_dir, f'valid.{args.target_lang}')

        # Validate preprocessed files
        missing_files = []
        for file in [src_file_bpe, tgt_file_bpe, valid_src_bpe, valid_tgt_bpe]:
            if not os.path.exists(file):
                missing_files.append(file)
        if missing_files:
            logging.error(f"Preprocessing failed: Output files not found: {missing_files}")
            sys.exit(1)

        # Load the new training dataset
        logging.info(f"Loading preprocessed training data: {src_file_bpe}, {tgt_file_bpe}")
        train_dataset = load_data(src_file=src_file_bpe, tgt_file=tgt_file_bpe, src_dict=src_dict, tgt_dict=tgt_dict)

        # Load the new validation dataset
        logging.info(f"Loading preprocessed validation data: {valid_src_bpe}, {valid_tgt_bpe}")
        valid_dataset = load_data(src_file=valid_src_bpe, tgt_file=valid_tgt_bpe, src_dict=src_dict, tgt_dict=tgt_dict)

        # Initialize DataLoader
        train_loader = DataLoader(
            train_dataset,
            num_workers=1,
            collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(
                RandomSampler(train_dataset, generator=torch.Generator().manual_seed(42)),
                batch_size=args.batch_size,
                drop_last=False
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
                logging.warning(f"Source: {src_dict.string(sample['src_tokens'].tolist()[0])}")
                logging.warning(f"Target: {tgt_dict.string(sample['tgt_tokens'].tolist()[0])}")

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()


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
        if epoch % args.save_interval == 0 and not args.no_save:
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


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("train.log", mode='a')
        ]
    )
    main(args)
