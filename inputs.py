# input:
#     [board position]

# output: (basically a move)
#     - src square
#     - dst square
#     - piece (rook, pawn, queen, etc.)


# training method:
#     - games from 2500+ rated players
#     - having the model try to learn moves made by the winner



import chess
from chess import pgn
import numpy as np
import gzip


PIECES_ORDER = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']  # Chess piece types
PIECES = {name: i for i, name in enumerate(PIECES_ORDER)}

def board_to_tensors(board):
    tensor_board = np.zeros((12, 8, 8), dtype=np.float16)

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

    tensor_extras = np.zeros(7, dtype=np.float16)
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
for i, src_file in enumerate('abcdefgh'):
    for j, src_rank in enumerate('12345678'):
        for k, dst_file in enumerate('abcdefgh'):
            for l, dst_rank in enumerate('12345678'):
                for piece_name in PIECES_ORDER:
                    # 128 for the board (8*8*2), 6 for the piece
                    t = np.zeros(128+6, dtype=np.float16)
                    src_index = i*8+j
                    dst_index = k*8+l
                    src_move = src_file + src_rank 
                    dst_move = dst_file + dst_rank
                    t[src_index] = 1.
                    t[64+dst_index] = 1.
                    piece_index = PIECES[piece_name]
                    t[128+piece_index] = 1.
                    cache[src_move+dst_move+piece_name] = t

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

#def move_to_tensor(move: chess.Move, board_after: chess.Board):
#    piece_name = chess.piece_name(board_after.piece_type_at(move.to_square))
#    return cache[move.uci()[:4] + piece_name]




if 1:
    input_board_tensors = []
    input_extras_tensors = []
    output_tensors = []
    num_games = 0
    
    f = gzip.open('output.chess.gz', 'rb')
    for line in f:
        moves = line.split(b",")
        winner = next(f).strip()
        if winner == b"D":
            continue
    
        to_move = b"W"
    
        board = chess.Board()
        for (i, move_san) in enumerate(moves):
            move_san = move_san.strip()
    
            # (was made by winner)
            use_as_input = to_move == winner and i < len(moves)
    
            if use_as_input:
                b, e = board_to_tensors(board)
                old_board = board.copy()
                input_board_tensors.append(b)
                input_extras_tensors.append(e)
    
            move = board.push_san(move_san.decode('utf-8'))
    
            if use_as_input:
                o = move_to_tensor(move, board)
                output_tensors.append(o)

            to_move = b"W" if to_move == b"B" else b"B"
    
        num_games += 1
    
        if num_games % 1000 == 0:
            print(f'processed {num_games} games already')
    
        # if num_games >= 10000:
        #     break
        if num_games >= 200_000:
            break
    
    
    print(f'loaded {len(input_board_tensors)} input  total')
    
        
    training_size = 100_000
    training_size = len(input_board_tensors)
    input_board_batch = np.stack(input_board_tensors[:training_size])
    input_extras_batch = np.stack(input_extras_tensors[:training_size])
    output_batch = np.stack(output_tensors[:training_size])
    # output_src_batch = torch.stack(output_src_tensors[:training_size])
    # output_dst_batch = torch.stack(output_dst_tensors[:training_size])
    
    # test_input = torch.stack(input_sequences[training_size:])
    # test_output = torch.stack(output_sequences[training_size:])

