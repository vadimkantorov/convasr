import os
import gc
import gzip
import csv
import time
import argparse
import importlib
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import warpctc_pytorch

import dataset
import decoder
import model
from model import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path')
parser.add_argument('--val-data-path', nargs = '+')
parser.add_argument('--sample-rate', type = int, default = 16000)
parser.add_argument('--window-size', type = float, default = 0.02)
parser.add_argument('--window-stride', type = float, default = 0.01)
parser.add_argument('--window', default = 'hann', choices = ['hann', 'hamming'])
parser.add_argument('--num-workers', type = int, default = 10)
parser.add_argument('--train-batch-size', type = int, default = 40)
parser.add_argument('--val-batch-size', type = int, default = 80)
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
parser.add_argument('--checkpoint')
parser.add_argument('--checkpoint-dir', default = 'data/checkpoints')
parser.add_argument('--model', default = 'Wav2LetterRu')
parser.add_argument('--tensorboard-log-dir', default = 'data/tensorboard')
parser.add_argument('--seed', default = 1)
parser.add_argument('--id', default = time.strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--max-norm', type = float, default = 100)
parser.add_argument('--lr', type = float, default = 5e-3)
parser.add_argument('--weight-decay', type = float, default = 1e-5)
parser.add_argument('--momentum', type = float, default = 0.5)
parser.add_argument('--nesterov', action = 'store_true')
args = parser.parse_args()

for set_seed in [torch.manual_seed] + ([torch.cuda.manual_seed_all] if args.device != 'cpu' else []):
    set_seed(args.seed)
tensorboard = torch.utils.tensorboard.SummaryWriter(args.tensorboard_log_dir)

lang = importlib.import_module(args.lang)
labels = dataset.Labels(lang.LABELS, preprocess_text = lang.preprocess_text, preprocess_word = lang.preprocess_word)

if args.train_data_path:
    train_dataset = dataset.SpectrogramDataset(args.train_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels)
    train_sampler = dataset.BucketingSampler(train_dataset, batch_size=args.train_batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, batch_sampler = train_sampler)

val_loaders = {os.path.basename(val_data_path) : torch.utils.data.DataLoader(dataset.SpectrogramDataset(val_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels), num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size) for val_data_path in args.val_data_path}

model = model.Speech2TextModel(getattr(model, args.model)(num_classes = len(labels.char_labels)))
if args.checkpoint:
    load_checkpoint(model, args.checkpoint)
model = torch.nn.DataParallel(model).to(args.device)

criterion = warpctc_pytorch.CTCLoss()
decoder = decoder.GreedyDecoder(labels.char_labels)

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov)

iteration = 0
for epoch in range(args.epochs if args.train_data_path else 1):
    if args.train_data_path:
        model.train()
        for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(train_loader):
            iteration += 1

            input_sizes = (input_percentages.cpu() * inputs.shape[-1]).int()
            logits, probs, output_sizes = model(inputs.to(args.device), input_sizes)
            loss = (criterion(logits.transpose(0, 1), targets, output_sizes.cpu(), target_sizes.cpu()) / len(inputs))
            print(f'epoch: {epoch:02d} iter: {iteration:09d} loss: {float(loss):.2f}')
            if (torch.isinf(loss) | torch.isnan(loss)).any():
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

    with torch.no_grad():
        model.eval()
        for val_dataset_name, val_loader in val_loaders.items():
            num_words, num_chars, val_wer_sum, val_cer_sum = 0, 0, 0.0, 0.0
            cer_ = []
            for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(val_loader):
                input_sizes = (input_percentages.cpu() * inputs.shape[-1]).int()
                logits, probs, output_sizes = model(inputs.to(args.device), input_sizes)
                decoded_output, _ = decoder.decode(probs, output_sizes)
                target_strings = decoder.convert_to_strings(dataset.unpack_targets(targets, target_sizes))
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer, cer, wer_ref, cer_ref = dataset.get_cer_wer(decoder, transcript, reference)
                    print(val_dataset_name, 'REF: ', reference)
                    print(val_dataset_name, 'HYP: ', transcript)
                    print()
                    val_wer_sum += wer
                    val_cer_sum += cer
                    num_words += wer_ref
                    num_chars += cer_ref
                    cer_.append(cer / cer_ref)
            cer_avg = torch.tensor(cer_).mean()
            wer_avg = val_wer_sum / num_words
            cer_avg_total = val_cer_sum / num_chars
            print(f'{val_dataset_name} | WER: {wer_avg:.02%} CER: {cer_avg_total:.02%} | {cer_avg:.02%}')
            tensorboard.add_scalars(args.id + '_' + val_dataset_name, dict(wer_avg = wer_avg, cer_avg = cer_avg), epoch)

        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok = True)
            save_checkpoint(model.module, args.checkpoint_dir, epoch, iteration)
