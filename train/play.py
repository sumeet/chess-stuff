#!/usr/bin/env python
import chess
import torch
from IPython.display import display as idisplay, clear_output
from train import model
from inputs import PIECES_ORDER, board_to_tensors
import sys


def eprint(*args):
    print(*args, file=sys.stderr)


def display(board):
    if 0:
        idisplay(board)
        return
    eprint(board.unicode(), file=sys.stderr)


# device = torch.device("cpu")
device = torch.device("cuda:0")
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])


def print_topk_values(output, k=10):
    source_output, dest_output, piece_output = output[:, :64], output[:, 64:128], output[:, 128:]

    # Get the indices and values of the top-k values
    source_probs, source_indices = torch.topk(source_output, k)
    dest_probs, dest_indices = torch.topk(dest_output, k)
    piece_probs, piece_indices = torch.topk(piece_output, 6)

    eprint("Top", k, "source squares:")
    for i in range(k):
        # Convert the index back into file and rank
        source_square_index = source_indices[0][i].item()
        source_prob = source_probs[0][i].item()
        source_file = 'abcdefgh'[source_square_index // 8]
        source_rank = '12345678'[source_square_index % 8]
        # Print the source square
        eprint("Square", i + 1, ":", source_file + source_rank, source_prob)

    eprint("Top", k, "destination squares:")
    for i in range(k):
        # Convert the index back into file and rank
        dest_square_index = dest_indices[0][i].item()
        dest_prob = dest_probs[0][i].item()
        dest_file = 'abcdefgh'[dest_square_index // 8]
        dest_rank = '12345678'[dest_square_index % 8]
        # Print the destination square
        eprint("Square", i + 1, ":", dest_file + dest_rank, dest_prob)

    eprint("Top pieces:")
    for i in range(6):
        # Get the piece
        piece_index = piece_indices[0][i].item()
        piece_prob = piece_probs[0][i].item()
        # Print the piece
        eprint("Piece", i + 1, ":", PIECES_ORDER[piece_index], piece_prob)


def get_move_from_top_src_and_dst(output):
    print_topk_values(output)

    source_output, dest_output, piece_output = output[:, :64], output[:, 64:128], output[:, 128:]

    # Get the index of the highest value in each half
    source_square_index = torch.argmax(source_output).item()
    dest_square_index = torch.argmax(dest_output).item()
    piece_index = torch.argmax(piece_output).item()

    return piece_and_move(source_square_index, dest_square_index, piece_index)


def piece_and_move(source_square_index, dest_square_index, piece_index):
    # Convert the indices back into file and rank
    source_file = 'abcdefgh'[source_square_index // 8]
    source_rank = '12345678'[source_square_index % 8]
    dest_file = 'abcdefgh'[dest_square_index // 8]
    dest_rank = '12345678'[dest_square_index % 8]
    # Return the move as a string
    return PIECES_ORDER[piece_index], source_file + source_rank + dest_file + dest_rank


board = chess.Board()

def find_move(board, piecename, movestr):
    for move in board.legal_moves:
        pieceat = board.piece_at(move.from_square)
        if move.uci()[:4] == movestr and chess.piece_name(pieceat.piece_type) == piecename:
            return move
    return None

def get_most_confident_legal_moves(output):
    print_topk_values(output)
    source_output, dest_output, piece_output = output[:, :64], output[:, 64:128], output[:, 128:]

    # Get the indices and values of the top-k values
    source_probs, source_indices = torch.topk(source_output, 64)
    dest_probs, dest_indices = torch.topk(dest_output, 64)
    piece_probs, piece_indices = torch.topk(piece_output, 6)

    source_probs = source_probs[0].cpu().detach().numpy()
    source_indices = source_indices[0].cpu().detach().numpy()
    dest_probs = dest_probs[0].cpu().detach().numpy()
    dest_indices = dest_indices[0].cpu().detach().numpy()
    piece_probs = piece_probs[0].cpu().detach().numpy()
    piece_indices = piece_indices[0].cpu().detach().numpy()

    probs_and_piecemoves = []
    for (source_prob, source_index) in zip(source_probs, source_indices):
        for (dest_prob, dest_index) in zip(dest_probs, dest_indices):
            for (piece_prob, piece_index) in zip(piece_probs, piece_indices):
                piece, move = piece_and_move(source_index, dest_index, piece_index)
                probs_and_piecemoves.append((source_prob + dest_prob + piece_prob, piece, move))

    legal_moves_strs = {move.uci()[:4] for move in board.legal_moves}
    acc = []
    printed = 0
    for (prob, piece, move) in probs_and_piecemoves:
        if move not in legal_moves_strs: continue

        move = find_move(board, piece, move)
        if not move:
            continue
        if printed < 10:
            eprint(f'found legal move {piece} {move} with prob {prob}')
            printed += 1
            acc.append((prob, piece, move.uci()[:4]))
    acc.sort(key=lambda x: x[0], reverse=True)
    return acc


def play():
    # Convert board state to model input
    input_tensor, input_extras = board_to_tensors(board)
    input_tensor = input_tensor.unsqueeze(0)
    input_extras = input_extras.unsqueeze(0)

    with torch.cuda.amp.autocast():
        output = model(input_tensor.to(device), input_extras.to(device))

    # Convert model output to move
    # piece, ai_move = get_move_from_top_src_and_dst(output)
    _, piece, ai_move = get_most_confident_legal_moves(output)[0]
    eprint(ai_move, piece)
    return ai_move


while True:
    [command, *args] = input().split()
    eprint(f'got {command}')
    match command:
        case 'uci':
            print('id name MyChessEngine')
            print('id author Me')
            print('uciok')
        case 'isready':
            print('readyok')
        case 'ucinewgame':
            board = chess.Board()
        case 'position':
            board = chess.Board()
            assert args[0] == 'startpos'
            if len(args) == 1:
                continue
            assert args[1] == 'moves'
            moves = args[2:]
            for move in moves:
                board.push(chess.Move.from_uci(move))
        case 'go':
            print('bestmove', play())
        case 'quit':
            break
        case unknown:
            eprint(f'Unknown command: {unknown}')
