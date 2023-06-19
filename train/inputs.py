# input:
#     [board position]
import gc
import multiprocessing
from itertools import islice, chain
import h5py

# output: (basically a move)
#     - src square
#     - dst square
#     - piece (rook, pawn, queen, etc.)


# training method:
#     - games from 2500+ rated players
#     - having the model try to learn moves made by the winner


import chess
import gzip
import torch

PIECES_ORDER = [
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "king",
]  # Chess piece types
PIECES = {name: i for i, name in enumerate(PIECES_ORDER)}


def board_to_tensors(board):
    tensor_board = torch.zeros((12, 8, 8), dtype=torch.float16)

    # Loop through all squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get piece type and color
            piece_type = chess.piece_name(piece.piece_type)
            color = piece.color

            # Calculate the coordinates
            row = square // 8
            col = square % 8

            # Calculate the depth index based on the piece type and color
            piece_index = PIECES[piece_type] + (6 if color == chess.BLACK else 0)

            # Place the piece in the tensor
            tensor_board[piece_index, row, col] = 1

    tensor_extras = torch.zeros(7, dtype=torch.float16)
    # Append castling rights
    tensor_extras[0] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor_extras[1] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor_extras[2] = int(board.has_kingside_castling_rights(chess.BLACK))
    tensor_extras[3] = int(board.has_queenside_castling_rights(chess.BLACK))

    # Append en passant square
    if board.ep_square is not None:
        tensor_extras[4] = board.ep_square
    else:
        tensor_extras[4] = 64

    # Append white's turn
    tensor_extras[5] = 1 if board.turn == chess.WHITE else 0

    # Append is in check
    tensor_extras[6] = 0 if board.is_check() else 1

    return (tensor_board, tensor_extras)


cache = {}
for i, src_file in enumerate("abcdefgh"):
    for j, src_rank in enumerate("12345678"):
        for k, dst_file in enumerate("abcdefgh"):
            for l, dst_rank in enumerate("12345678"):
                for piece_name in PIECES_ORDER:
                    # 128 for the board (8*8*2), 6 for the piece
                    t = torch.zeros(128 + 6, dtype=torch.float16)
                    src_index = i * 8 + j
                    dst_index = k * 8 + l
                    src_move = src_file + src_rank
                    dst_move = dst_file + dst_rank
                    t[src_index] = 1.0
                    t[64 + dst_index] = 1.0
                    piece_index = PIECES[piece_name]
                    t[128 + piece_index] = 1.0
                    cache[src_move + dst_move + piece_name] = t


def move_to_tensor(move: chess.Move, board_after: chess.Board):
    piece_name = chess.piece_name(board_after.piece_type_at(move.to_square))
    return cache[move.uci()[:4] + piece_name]


# cache = {}
# for i, src_file in enumerate('abcdefgh'):
#     for j, src_rank in enumerate('12345678'):
#         for k, dst_file in enumerate('abcdefgh'):
#             for l, dst_rank in enumerate('12345678'):
#                 for piece_name in PIECES_ORDER:
#                     piece = PIECES[piece_name]
#                     # 2 for the board, 1 for the piece
#                     t = torch.zeros(3, dtype=torch.float16)
#                     t[0] = i*8+j
#                     t[1] = k*8+l
#                     t[2] = PIECES[piece_name]
#                     cache[src_file+src_rank+dst_file+dst_rank+piece_name] = t

# def move_to_tensor(move: chess.Move, board_after: chess.Board):
#    piece_name = chess.piece_name(board_after.piece_type_at(move.to_square))
#    return cache[move.uci()[:4] + piece_name]

def process_line_chunk(line_chunk):
    i, line_chunk = line_chunk

    input_board = []
    input_extras = []
    output = []
    for line_pair in line_chunk:
        boards, extras, outputs = process_line_pair(line_pair)
        input_board.extend(boards)
        input_extras.extend(extras)
        output.extend(outputs)
    with LOCK:
        with h5py.File(OUTPUT_H5_FILENAME, "a") as f:
            [input_board_ds, input_extras_ds, output_ds] = [
                f[ds_name] for ds_name in ["input_board", "input_extras", "output"]
            ]
            current_size = input_board_ds.shape[0]
            new_size = current_size + len(input_board)

            for ds, data in zip(
                    [input_board_ds, input_extras_ds, output_ds],
                    [input_board, input_extras, output]):
                stacked_data = torch.stack(data)
                ds.resize((new_size,) + stacked_data.shape[1:])
                ds[current_size:new_size, ...] = stacked_data

    print(f"Processed boards: {i * 10_000}")


def process_line_pair(line_pair):
    input_board_tensors = []
    input_extras_tensors = []
    output_tensors = []

    moves, winner = line_pair
    moves = moves.split(b",")
    winner = winner.strip()

    if winner == b"D":
        return [], [], []

    to_move = b"W"

    board = chess.Board()
    for i, move_san in enumerate(moves):
        move_san = move_san.strip()

        use_as_input = to_move == winner and i < len(moves)

        if use_as_input:
            b, e = board_to_tensors(board)
            input_board_tensors.append(b)
            input_extras_tensors.append(e)

        move = board.push_san(move_san.decode("utf-8"))

        if use_as_input:
            o = move_to_tensor(move, board)
            output_tensors.append(o)

        to_move = b"W" if to_move == b"B" else b"B"

    return input_board_tensors, input_extras_tensors, output_tensors


LOCK = multiprocessing.Lock()


def chunks(iterable, chunk_size):
    it = iter(iterable)
    while c := list(islice(it, chunk_size)):
        yield c


OUTPUT_H5_FILENAME = "input_ALL.h5"

if __name__ == '__main__':
    with gzip.open("output.chess.gz", "rb") as f:
        line_pairs = ((line1, line2) for line1, line2 in zip(f, f))
        #training_size = 100_000
        #line_pairs = islice(line_pairs, training_size)

        with h5py.File(OUTPUT_H5_FILENAME, "w") as f:
            input_board_ds = f.create_dataset(
                "input_board", (0, 12, 8, 8), chunks=(10_000, 12, 8, 8), maxshape=(None, 12, 8, 8), dtype="float16", compression="gzip"
            )
            input_extras_ds = f.create_dataset(
                "input_extras", (0, 7), chunks=(10_000, 7), maxshape=(None, 7), dtype="float16", compression="gzip"
            )
            output_ds = f.create_dataset(
                "output", (0, 134), chunks=(10_000, 134), maxshape=(None, 134), dtype="float16", compression="gzip"
            )

        with multiprocessing.Pool() as p:
            chunk_size = 5_000
            pair_chunks = chunks(line_pairs, chunk_size)
            results = p.imap_unordered(process_line_chunk, enumerate(pair_chunks))

            for i, _ in enumerate(results):
                print(f"processed {chunk_size * (i + 1)} games")

