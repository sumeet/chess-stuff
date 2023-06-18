from functools import cached_property, cache

import torch
import gzip
import h5py
from torch.utils.data import TensorDataset, Dataset, random_split, Subset


def random_split_perc(dataset, percentages):
    total_data_len = len(dataset)
    split_lengths = [int(total_data_len * p) for p in percentages]
    # If there is any difference in sum, assign it to the last length
    sum_of_lengths = sum(split_lengths)
    if sum_of_lengths != total_data_len:
        split_lengths[-1] += total_data_len - sum_of_lengths
    return random_split(dataset, lengths=split_lengths)


def ordered_split_perc(dataset, percentages):
    total_data_len = len(dataset)
    split_lengths = [int(total_data_len * p) for p in percentages]
    # If there is any difference in sum, assign it to the last length
    sum_of_lengths = sum(split_lengths)
    if sum_of_lengths != total_data_len:
        split_lengths[-1] += total_data_len - sum_of_lengths
    return [Subset(dataset, range(start, start + length))
            for start, length in zip([0] + split_lengths[:-1], split_lengths)]


import contextlib

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# output format:
# [[1/64 src square][1/64 dst square][1/6 piece]]

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 128*8*8 is the size of the flattened conv layer output,
        # and 7 is the size of the extra features tensor
        self.fc1 = nn.Linear(128 * 8 * 8 + 7, 256)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(256, 128 + 6)

    def forward(self, board_tensor, extra_features):
        out = F.relu(self.conv1(board_tensor))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)  # Flatten tensor
        out = torch.cat((out, extra_features), dim=1)  # Concatenate extra features
        out = self.relu(self.fc1(out))
        out = self.fc_out(out)
        return out

    def run(self, dataloader, is_train):
        print("Running model, is_train:", is_train)

        model = self.train() if is_train else self.eval()
        tot_loss = 0
        num_iterations = 0

        ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with ctx:
            for inputs_board, inputs_extras, targets in dataloader:
                print("batch: inputs_board.shape", inputs_board.shape)

                inputs_board = inputs_board.to(device)
                inputs_extras = inputs_extras.to(device)
                targets = targets.to(device)

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs_board, inputs_extras)

                    targets_source, targets_dest, targets_square = targets[:, :64], targets[:, 64:128], targets[:, 128:]
                    outputs_source, outputs_dest, outputs_square = outputs[:, :64], outputs[:, 64:128], outputs[:, 128:]
                    loss_source = loss_fn(outputs_source, targets_source)
                    loss_dest = loss_fn(outputs_dest, targets_dest)
                    loss_square = loss_fn(outputs_square, targets_square)
                    # Compute total loss
                    loss = loss_source + loss_dest + loss_square
                    tot_loss += loss
                    num_iterations += 1

                if is_train:
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return tot_loss / num_iterations


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChessMovePredictor()
model = model.to(device)
# loss_fn = nn.NLLLoss()
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import time

if __name__ == '__main__':
    if 0:
        print("Loading data from input.h5 into memory...")
        stop = 50_000

        with h5py.File('input.h5', "r") as f:
            input_board_batch = torch.from_numpy(f["input_board"][:stop])
            print("input_board_batch.shape", input_board_batch.shape)
            input_extras_batch = torch.from_numpy(f["input_extras"][:stop])
            print("input_extras_batch.shape", input_extras_batch.shape)
            output_batch = torch.from_numpy(f["output"][:stop])
            print("output_batch.shape", output_batch.shape)

        dataset = TensorDataset(input_board_batch,
                                input_extras_batch,
                                output_batch)
        print("... Done")

    else:
        class SimpleHDF5Dataset(Dataset):
            def __init__(self, file_path):
                self.file_path = file_path
                self.file = self.open_file()

            def reopen_file(self):
                self.file = self.open_file()

            def open_file(self):
                return h5py.File(self.file_path, "r")

            def __len__(self):
                return len(self.file["input_board"])

            def __getitem__(self, idx):
                input_board = self.file["input_board"][idx]
                input_extras = self.file["input_extras"][idx]
                output = self.file["output"][idx]
                return input_board, input_extras, output


        class CachedHDF5Dataset(Dataset):
            KEYS = ["input_board", "input_extras", "output"]

            def __init__(self, file_path):
                self.file_path = file_path
                self.worker_cache = {}
                self.chunk_size = None
                self.chunk_start = None
                self.chunk_end = None

            @cache
            def __len__(self):
                with self.open_file() as f:
                    return len(f[self.KEYS[0]])

            def __getitem__(self, idx):
                # our_idx = idx - self.chunk_start
                # if our_idx < 0 or our_idx >= self.chunk_size:
                #     raise Exception(f"Index {idx} out of my range ({self.chunk_start}, {self.chunk_end})")
                # Chatgpt says the math just works out
                our_idx = idx % self.chunk_size
                return tuple(self.worker_cache[key][our_idx] for key in self.KEYS)

            def read_into_worker_cache(self):
                if self.worker_cache:
                    print('trying to read into worker cache but already read')
                    return

                if not (worker_info := torch.utils.data.get_worker_info()):
                    raise Exception("No worker info")
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                print(f'worker_id {worker_id} reading into worker cache')
                assert worker_id < num_workers
                self.chunk_size = len(self) // num_workers
                print(f'setting chunk_size to {self.chunk_size} (num_workers: {num_workers}')
                self.chunk_start = self.chunk_size * worker_id
                self.chunk_end = self.chunk_size * (worker_id + 1) if worker_id < num_workers - 1 else len(self)
                with self.open_file() as f:
                    self.worker_cache = {key: f[key][self.chunk_start:self.chunk_end] for key in self.KEYS}
                print('done reading into worker cache')

            def open_file(self):
                return h5py.File(self.file_path, "r")

        dataset = CachedHDF5Dataset("./input_100k.h5")

    num_epochs = 10_000
    # training_set, validation_set = split_perc(dataset, [0.8, 0.2])
    training_set, validation_set = ordered_split_perc(dataset, [0.8, 0.2])
    #training_set, validation_set, _ = ordered_split_perc(dataset, [0.01, 0.01, 0.98])

    training_dataloader = DataLoader(training_set,
                                     shuffle=False,
                                     drop_last=True,
                                     batch_size=10_000,
                                     persistent_workers=True,
                                     num_workers=cpu_count() - 1,
                                     prefetch_factor=50,
                                     worker_init_fn=lambda *x: dataset.read_into_worker_cache(),
                                     pin_memory=True)
    validation_dataloader = DataLoader(validation_set,
                                       shuffle=False,
                                       drop_last=True,
                                       batch_size=10_000,
                                       persistent_workers=True,
                                       num_workers=cpu_count() - 1,
                                       prefetch_factor=50,
                                       worker_init_fn=lambda *x: dataset.read_into_worker_cache(),
                                       pin_memory=True)

    print('starting the epochs')
    for epoch in range(num_epochs):
        epoch_start = time.time()

        print('running training')
        training_loss = model.run(training_dataloader, is_train=True)
        print('running validation')
        validation_loss = model.run(validation_dataloader, is_train=False)

        if epoch % 5 == 0:
            filename = 'checkpoint.pt'
            print(f'checkpointing torch model & optimizer to {filename}')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training_loss': training_loss,
                'validation_loss': validation_loss,
            }, filename)

        time_elapsed = time.time() - epoch_start
        print('Epoch [{}/{}], Training loss: {:.4f} Validation loss: {:.4f} ({} seconds)'
              .format(epoch + 1, num_epochs, training_loss, validation_loss, time_elapsed))
