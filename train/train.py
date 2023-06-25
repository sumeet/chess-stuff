import torch
import h5py
from torch.utils.data import TensorDataset, Dataset, random_split, Subset, IterableDataset


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


# output format:
# [[1/64 src square][1/64 dst square][1/6 piece]]

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # 128*8*8 is the size of the flattened conv layer output,
        # and 7 is the size of the extra features tensor
        self.fc1 = nn.Linear(128 * 8 * 8 + 7, 256)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(256, 128 + 6)

    def forward(self, board_tensor, extra_features):
        out1 = self.conv1(board_tensor)
        out = F.relu(out1 + self.conv2(out1))  # add residual connection
        out2 = self.conv3(out)
        out = F.relu(out2 + self.conv4(out2))  # add residual connection
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

from multiprocessing import cpu_count, Value
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


        class CachedHDF5Dataset(IterableDataset):

            AVAIL_MEMORY = 30 * 1024 * 1024 * 1024
            KEYS = ["input_board", "input_extras", "output"]

            seek_position = Value('i', 0)

            def __init__(self, file_path):
                self.file_path = file_path
                self.chunk_size = None
                self.total_len = None

            def __iter__(self):
                with self.seek_position.get_lock():
                    current_seek_position = self.seek_position.value
                    chunk_start = current_seek_position
                    chunk_end = current_seek_position + self.chunk_size
                    self.seek_position.value = chunk_end % self.total_len

                chunk_groups = []
                if chunk_end > self.total_len:
                    chunk_groups.append((chunk_start, self.total_len))
                    chunk_groups.append((0, chunk_end % self.total_len))
                else:
                    chunk_groups.append((chunk_start, chunk_end))

                print(f'loading chunk groups: {chunk_groups}')

                with self.open_file() as f:
                    for chunk_start, chunk_end in chunk_groups:
                        this_batch = [f[key][chunk_start:chunk_end] for key in self.KEYS]
                        for i in range(len(this_batch[0])):
                            yield tuple([batch[i] for batch in this_batch])

            def worker_init(self):
                worker_info = torch.utils.data.get_worker_info()
                assert worker_info is not None
                num_workers = worker_info.num_workers

                with self.open_file() as f:
                    self.total_len = len(f[self.KEYS[0]])
                    sample_memory = 0
                    for key in self.KEYS:
                        sample_memory += f[key][0].nbytes
                self.chunk_size = self.AVAIL_MEMORY // (sample_memory * num_workers)

            def open_file(self):
                return h5py.File(self.file_path, "r")


        dataset = CachedHDF5Dataset("./input_ALL.h5")

    num_epochs = 10_000
    # training_set, validation_set = split_perc(dataset, [0.8, 0.2])
    #training_set, validation_set = ordered_split_perc(dataset, [0.8, 0.2])
    # training_set, validation_set, _ = ordered_split_perc(dataset, [0.01, 0.01, 0.98])
    training_set = dataset

    training_dataloader = DataLoader(training_set,
                                     shuffle=False,
                                     drop_last=True,
                                     batch_size=10_000,
                                     persistent_workers=True,
                                     num_workers=cpu_count() - 1,
                                     prefetch_factor=50,
                                     worker_init_fn=lambda *x: dataset.worker_init(),
                                     pin_memory=True)
    # validation_dataloader = DataLoader(validation_set,
    #                                    shuffle=False,
    #                                    drop_last=True,
    #                                    batch_size=10_000,
    #                                    persistent_workers=True,
    #                                    num_workers=cpu_count() - 1,
    #                                    prefetch_factor=50,
    #                                    worker_init_fn=lambda *x: dataset.read_into_worker_cache(),
    #                                    pin_memory=True)

    print('starting the epochs')
    training_losses = []
    for epoch in range(num_epochs):
        epoch_start = time.time()

        print('running training')
        training_loss = model.run(training_dataloader, is_train=True)
        training_losses.append(training_loss.item())
        print('running validation')
        # validation_loss = model.run(validation_dataloader, is_train=False)
        validation_loss = float('nan')

        if True:#epoch % 4 == 0:
            filename = 'checkpoint.pt'
            print(f'checkpointing torch model & optimizer to {filename}')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training_loss': training_loss,
                'training_losses': training_losses,
                'validation_loss': validation_loss,
            }, filename)

        time_elapsed = time.time() - epoch_start
        print('Epoch [{}/{}], Training loss: {:.4f} Validation loss: {:.4f} ({} seconds)'
              .format(epoch + 1, num_epochs, training_loss, validation_loss, time_elapsed))
