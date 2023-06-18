from functools import cached_property

import torch
import gzip
import h5py
from torch.utils.data import TensorDataset, Dataset, random_split


def split_perc(dataset, percentages):
    total_data_len = len(dataset)
    split_lengths = [int(total_data_len * p) for p in percentages]
    # If there is any difference in sum, assign it to the last length
    sum_of_lengths = sum(split_lengths)
    if sum_of_lengths != total_data_len:
        split_lengths[-1] += total_data_len - sum_of_lengths
    return random_split(dataset, lengths=split_lengths)


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

            def __init__(self, file_path, max_memory_size_bytes, cache_chunk_len, max_chunk_hits, worker_id=None, num_workers=None):
                self.file_path = file_path

                self.worker_id = worker_id
                self.num_workers = num_workers

                self.file = h5py.File(self.file_path, 'r')
                self.total_length = len(self.file['input_board'])

                self.max_memory_size_bytes = max_memory_size_bytes
                self.cache_chunk_len = cache_chunk_len
                self.max_chunk_hits = max_chunk_hits

                self.cache = {}
                self.next_segment_idx = 0

                # Calculate how many chunks can fit in max_memory_size_bytes
                single_item_size_bytes = (self.file['input_board'][0].nbytes
                                          + self.file['input_extras'][0].nbytes
                                          + self.file['output'][0].nbytes)
                self.max_chunks = (self.max_memory_size_bytes - 1) // (single_item_size_bytes * self.cache_chunk_len) + 1

            def __len__(self):
                single_item_size_bytes = (self.file['input_board'][0].nbytes +
                                          self.file['input_extras'][0].nbytes +
                                          self.file['output'][0].nbytes)
                max_derivable_items = self.max_memory_size_bytes // single_item_size_bytes
                return min(self.total_length, max_derivable_items)

            def load_next_chunk(self):
                if self.next_segment_idx * self.cache_chunk_len >= self.total_length:
                    self.next_segment_idx = 0

                chunk_start = self.next_segment_idx * self.cache_chunk_len
                chunk_end = chunk_start + self.cache_chunk_len

                def wrap_around_slice(field):
                    if chunk_end > self.total_length:
                        overflow_end = chunk_end - self.total_length
                        return torch.cat([
                            torch.tensor(self.file[field][chunk_start:self.total_length]),
                            torch.tensor(self.file[field][:overflow_end])
                        ])
                    else:
                        return torch.tensor(self.file[field][chunk_start:chunk_end])

                chunk = {
                    'input_board': wrap_around_slice('input_board'),
                    'input_extras': wrap_around_slice('input_extras'),
                    'output': wrap_around_slice('output'),
                    'hits': 0  # Hits should be an integer
                }
                self.next_segment_idx += 1
                return chunk

            def fetch(self, idx):
                chunk_idx = idx // self.cache_chunk_len
                local_idx = idx % self.cache_chunk_len

                if chunk_idx not in self.cache:
                    print(f'seeding chunk {chunk_idx} into cache')
                    self.cache[chunk_idx] = self.load_next_chunk()

                chunk = self.cache[chunk_idx]
                chunk['hits'] += 1
                if chunk['hits'] >= self.max_chunk_hits:
                    print(f'deleting chunk {chunk_idx} from cache, and loading next chunk')
                    chunk = self.cache[chunk_idx] = self.load_next_chunk()

                return chunk['input_board'][local_idx], chunk['input_extras'][local_idx], chunk['output'][local_idx]

            def __getitem__(self, idx):
                return self.fetch(idx % self.total_length)

        dataset = CachedHDF5Dataset("./input.h5", max_memory_size_bytes=50_000_000_000, cache_chunk_len=10_000, max_chunk_hits=10_000)

    num_epochs = 10_000
    training_set, validation_set = split_perc(dataset, [0.8, 0.2])

    training_dataloader = DataLoader(training_set,
                                     shuffle=True,
                                     batch_size=10_000,
                                     persistent_workers=True,
                                     num_workers=cpu_count() - 1,
                                     prefetch_factor=50,
                                     pin_memory=True)
    validation_dataloader = DataLoader(validation_set,
                                       shuffle=True,
                                       batch_size=10_000,
                                       persistent_workers=True,
                                       num_workers=cpu_count() - 1,
                                       prefetch_factor=50,
                                       pin_memory=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        training_loss = model.run(training_dataloader, is_train=True)
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
