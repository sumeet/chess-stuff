import torch
import torch.distributed as dist
import h5py
from torch.utils.data import TensorDataset, Dataset, random_split, Subset, IterableDataset
#from multiprocessing import cpu_count, Value
from torch.multiprocessing import cpu_count, Value
from torch.utils.data import DataLoader
import time
import os
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler



class CachedHDF5Dataset(IterableDataset):

    AVAIL_MEMORY = 30 * 1024 * 1024 * 1024
    KEYS = ["input_board", "input_extras", "output"]

    seek_position = Value('i', 0)

    def __init__(self, file_path, total_len=None):
        self.file_path = file_path
        self.chunk_size = None
        if total_len:
            self.total_len = total_len
        else:
            with self.open_file() as f:
                self.total_len = len(f[self.KEYS[0]])

        self.should_keep_in_memory = False
        self.kept_in_memory = None

    def __iter__(self):
        if self.should_keep_in_memory and self.kept_in_memory:
            for i in range(len(self.kept_in_memory[0])):
                yield tuple([batch[i] for batch in self.kept_in_memory])
            return

        # with self.seek_position.get_lock():
        #     current_seek_position = self.seek_position.value
        #     chunk_start = current_seek_position
        #     chunk_end = current_seek_position + self.chunk_size
        #     self.seek_position.value = chunk_end % self.total_len
        #
        # chunk_groups = []
        # if chunk_end > self.total_len:
        #     chunk_groups.append((chunk_start, self.total_len))
        #     chunk_groups.append((0, chunk_end % self.total_len))
        # else:
        #     chunk_groups.append((chunk_start, chunk_end))
        #
        # print(f'loading chunk groups: {chunk_groups}')
        chunk_groups = [(self.chunk_start, self.chunk_end)]

        with self.open_file() as f:
            for chunk_start, chunk_end in chunk_groups:
                # modify chunk_start and chunk_end to take into account rank and world_count:
                rank = self.rank
                world_size = self.world_size
                chunk_start = chunk_start + rank * (chunk_end - chunk_start) // world_size
                chunk_end = chunk_start + (chunk_end - chunk_start) // world_size
                print(f'loading chunk in {rank} / {world_size}: {chunk_start} - {chunk_end}')

                this_batch = [f[key][chunk_start:chunk_end] for key in self.KEYS]
                if self.should_keep_in_memory:
                    self.kept_in_memory = this_batch
                for i in range(len(this_batch[0])):
                    yield tuple([batch[i] for batch in this_batch])

    def worker_init(self, *args):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        num_workers = worker_info.num_workers

        with self.open_file() as f:
            sample_memory = 0
            for key in self.KEYS:
                sample_memory += f[key][0].nbytes

        if sample_memory * self.total_len <= self.AVAIL_MEMORY:
            self.chunk_size = self.total_len // num_workers * 2
            self.should_keep_in_memory = True
        else:
            self.chunk_size = self.AVAIL_MEMORY // (sample_memory * num_workers)

        self.chunk_start = self.chunk_size * worker_info.id
        self.chunk_end = self.chunk_start + self.chunk_size

    def open_file(self):
        return h5py.File(self.file_path, "r")




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
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # N*8*8 is the size of the flattened conv layer output,
        # and 7 is the size of the extra features tensor
        self.fc1 = nn.Linear(256 * 8 * 8 + 7, 256)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(256, 128 + 6)

    def forward(self, board_tensor, extra_features):
        out1 = self.conv1(board_tensor)
        out = F.relu(out1 + self.conv2(out1))  # add residual connection
        out2 = self.conv3(out)
        out = F.relu(out2 + self.conv4(out2))  # add residual connection
        out3 = self.conv5(out)
        out = F.relu(out3 + self.conv6(out3))  # add residual connection

        out = out.view(out.size(0), -1)  # Flatten tensor
        out = torch.cat((out, extra_features), dim=1)  # Concatenate extra features
        out = self.relu(self.fc1(out))
        out = self.fc_out(out)
        return out

    @staticmethod
    def run(model, dataloader, is_train, rank):
        print("Running model, is_train:", is_train)

        model = model.train() if is_train else model.eval()
        tot_loss = 0
        num_iterations = 0

        ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with ctx:
            for inputs_board, inputs_extras, targets in dataloader:
                inputs_board = inputs_board.to(rank)
                inputs_extras = inputs_extras.to(rank)
                targets = targets.to(rank)

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs_board, inputs_extras)

                    targets_source, targets_dest, targets_piece = targets[:, :64], targets[:, 64:128], targets[:, 128:]
                    outputs_source, outputs_dest, outputs_piece = outputs[:, :64], outputs[:, 64:128], outputs[:, 128:]
                    loss_source = loss_fn(outputs_source, targets_source)
                    loss_dest = loss_fn(outputs_dest, targets_dest)
                    loss_piece = loss_fn(outputs_piece, targets_piece)
                    # Compute total loss
                    loss = loss_source + loss_dest + loss_piece
                    tot_loss += loss
                    num_iterations += 1

                if is_train:
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return tot_loss / num_iterations


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChessMovePredictor()
# model = model.to(device)
# loss_fn = nn.NLLLoss()
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')

    if 0:
        print("Loading data from input_ALL.h5 into memory...")
        total_len = 1_000_000

        start = rank * total_len // world_size
        stop = (rank + 1) * total_len // world_size

        with h5py.File('input_ALL.h5', "r") as f:
            input_board_batch = torch.from_numpy(f["input_board"][start:stop])
            print("input_board_batch.shape", input_board_batch.shape)
            input_extras_batch = torch.from_numpy(f["input_extras"][start:stop])
            print("input_extras_batch.shape", input_extras_batch.shape)
            output_batch = torch.from_numpy(f["output"][start:stop])
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


        dataset = CachedHDF5Dataset("./input_ALL.h5", total_len=5_000_000)

    num_epochs = 10_000
    # training_set, validation_set = split_perc(dataset, [0.8, 0.2])
    #training_set, validation_set = ordered_split_perc(dataset, [0.8, 0.2])
    # training_set, validation_set, _ = ordered_split_perc(dataset, [0.01, 0.01, 0.98])
    training_set = dataset
    dataset.rank = rank
    dataset.world_size = world_size
    # sampler = DistributedSampler(training_set)

    training_dataloader = DataLoader(training_set,
                                     shuffle=False,
                                     drop_last=True,
                                     batch_size=10_000,
                                     persistent_workers=True,
                                     num_workers=cpu_count() - 1,
                                     # prefetch_factor=50,
                                     worker_init_fn=dataset.worker_init,
                                     pin_memory=True)
    rank = dist.get_rank()
    model = ChessMovePredictor()
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
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
        # sampler.set_epoch(epoch)

        print('running training')
        training_loss = ChessMovePredictor.run(model, training_dataloader, is_train=True, rank=rank)
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


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    import torch.multiprocessing as mp
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)