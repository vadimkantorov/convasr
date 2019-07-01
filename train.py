import gc
import time
import argparse
import torch
import torch.utils.data
import torch.utils.tensorboard

import data.dataset
import model
import decoder
from warpctc_pytorch import CTCLoss

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', default = '../open_stt_splits/audiobooks_train.csv.gz')
parser.add_argument('--val-data-path', default = '../open_stt_splits/audiobooks_val.csv.gz')
parser.add_argument('--sample-rate', type = int, default = 16000)
parser.add_argument('--window-size', type = float, default = 0.2)
parser.add_argument('--window-stride', type = float, default = 0.1)
parser.add_argument('--window', default = 'hann', choices = ['hann', 'hamming'])
parser.add_argument('--num-workers', type = int, default = 1)
parser.add_argument('--lr', type = float, default = 3e-4)
parser.add_argument('--weight-decay', type = float, default = 0.0)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--train-batch-size', type = int, default = 40)
parser.add_argument('--val-batch-size', type = int, default = 80)
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
parser.add_argument('--checkpoint')
parser.add_argument('--checkpoint-dir', default = 'data/checkpoints')
parser.add_argument('--model', default = 'Wav2LetterVanilla')
args = parser.parse_args()

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force = True) # https://github.com/pytorch/pytorch/issues/22131
    labels = data.labels.RU_LABELS
    num_classes = len(labels)

    train_dataset = data.dataset.SpectrogramDataset(sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, data_path = args.train_data_path, labels = labels)
    train_sampler = data.dataset.BucketingSampler(train_dataset, batch_size=args.train_batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = data.dataset.collate_fn, pin_memory = True, batch_sampler = train_sampler)

    val_dataset = data.dataset.SpectrogramDataset(sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, data_path = args.val_data_path, labels = labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers = args.num_workers, collate_fn = data.dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size)

    device = torch.device(args.device)

    model = model.Speech2TextModel(getattr(model, args.model)(num_classes))
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    criterion = CTCLoss()
    decoder = decoder.GreedyDecoder(labels)
    
    print('Dataset loaded')
    for epoch in range(args.epochs):
        for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(train_loader):
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            logits, probs, output_sizes = model(inputs.to(device), input_sizes.to(device))

            loss = criterion(logits.transpose(0, 1), targets, output_sizes.cpu(), target_sizes) # logits -> TxNxH
            loss = loss / len(inputs) 
            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iteration', i, 'loss:', float(loss))

        num_words, num_chars, val_wer_sum, val_cer_sum = 0, 0, 0.0, 0.0

        for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(train_loader):
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            logits, probs, output_sizes = model(inputs.to(device), input_sizes.to(device))
            decoded_output, _ = decoder.decode(probs, output_sizes)
            target_strings = decoder.convert_to_strings(data.dataset.unpack_targets(targets, target_sizes))
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer, cer, wer_ref, cer_ref = data.dataset.get_cer_wer(decoder, transcript, reference)
                val_wer_sum += wer
                val_cer_sum += cer
                num_words += wer_ref
                num_chars += cer_ref
        print('WER: {:.02%} CER: {:.02%}'.format(val_wer_sum / num_words, val_cer_sum / num_chars))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model.save_checkpoint(arg.checkpoint_dir)
