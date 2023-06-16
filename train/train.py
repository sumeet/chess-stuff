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
        self.fc1 = nn.Linear(128*8*8 + 7, 256)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(256, 128+6)

    def forward(self, board_tensor, extra_features):
        out = F.relu(self.conv1(board_tensor))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)  # Flatten tensor
        out = torch.cat((out, extra_features), dim=1)  # Concatenate extra features
        out = self.relu(self.fc1(out))
        out = self.fc_out(out)
        return out

    def run(self, dataloader, is_train):
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
#loss_fn = nn.NLLLoss() 
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import time

if __name__ == '__main__':
    if 1:
        stop = 50_000

        with h5py.File('input.h5', "r") as f:
            input_board_batch = torch.from_numpy(f["input_board"][:stop])
            input_extras_batch = torch.from_numpy(f["input_extras"][:stop])
            output_batch = torch.from_numpy(f["output"][:stop])

        dataset = TensorDataset(input_board_batch,
                                input_extras_batch,
                                output_batch)

    else:
        class HDF5Dataset(Dataset):
            def __init__(self, file_path):
                self.file_path = file_path
                self.file = h5py.File(self.file_path, "r", rdcc_nslots=1991, rdcc_nbytes=16_000 * (1024 ** 3))

            def __len__(self):
                return len(self.file["input_board"])
    #            with h5py.File(self.file_path, "r") as file:
    #                return len(file["input_board"])

            def __getitem__(self, idx):
        #        with h5py.File(self.file_path, "r") as f:
                input_board = self.file["input_board"][idx]
                input_extras = self.file["input_extras"][idx]
                output = self.file["output"][idx]
                return input_board, input_extras, output

        dataset = HDF5Dataset("./input.h5")


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
            .format(epoch+1, num_epochs, training_loss, validation_loss, time_elapsed))

