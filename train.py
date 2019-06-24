import gc
import time
import argparse
import torch
import torch.utils.data
#import torch.utils.tensorboard

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
parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
args = parser.parse_args()

audio_conf = dict(
	sample_rate = args.sample_rate,
	window_size = args.window_size,
	window_stride = args.window_stride,
	window = args.window
)

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force = True) # https://github.com/pytorch/pytorch/issues/22131

    labels = data.labels.RU_LABELS
    train_dataset = data.dataset.SpectrogramDataset(audio_conf = audio_conf, data_path = args.train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = data.dataset.collate_fn, pin_memory = True, shuffle = True)
    num_classes = len(labels)

    val_dataset = data.dataset.SpectrogramDataset(audio_conf = audio_conf, data_path = args.val_data_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers = args.num_workers, collate_fn = data.dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size)

    device = torch.device(args.device)

    model = model.Speech2TextModel(model.Wav2LetterVanilla(num_classes))
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    criterion = CTCLoss()
    decoder = decoder.GreedyDecoder(labels)

    for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(train_loader):
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        logits, probs, output_sizes = model(inputs.to(device), input_sizes.to(device))
        decoded_output, _ = decoder.decode(probs, output_sizes)
        target_strings = decoder.convert_to_strings(data.dataset.unpack_targets(targets, target_sizes))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer, cer, wer_ref, cer_ref = data.dataset.get_cer_wer(decoder, transcript, reference)
            print('TRANSCRIPT:', transcript, 'REF:', reference, 'CER:', cer)

        logits = logits.transpose(0, 1)  # TxNxH
        loss = criterion(logits, targets, output_sizes.cpu(), target_sizes)
        loss = loss / len(inputs)  # average the loss by minibatch
        loss = loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
