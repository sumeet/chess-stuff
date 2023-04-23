use shakmaty::{Chess, Color, Move, Position, Role};
use std::io::Write;
use std::io::{self, BufRead};
use std::process::exit;
use vampirc_uci::{parse_one, UciMessage, UciSquare};

struct State {
    pos: Chess,
    white_sum: u8,
    black_sum: u8,
}

fn sq(uci_square: UciSquare) -> shakmaty::Square {
    let file = shakmaty::File::from_char(uci_square.file).unwrap();
    let rank = shakmaty::Rank::from_char((uci_square.rank + '0' as u8) as char).unwrap();
    shakmaty::Square::from_coords(file, rank)
}

impl State {
    fn init() -> Self {
        let pos = Chess::default();
        // calculate sum of piece values for white and black, separately
        let mut white_sum = 0;
        let mut black_sum = 0;
        let by_color = pos.board().material();
        let by_role_white = by_color.get(Color::White);
        let by_role_black = by_color.get(Color::Black);
        for role in Role::ALL {
            let pv = piece_value(role);
            white_sum += by_color.get(Color::White).get(role) * pv;
            black_sum += by_color.get(Color::Black).get(role) * pv;
        }
        Self {
            pos,
            white_sum,
            black_sum,
        }
    }
}

fn main() {
    let mut state = State::init();
    let mut log_file = std::fs::File::create("/tmp/log.txt").unwrap();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap());
        match msg {
            UciMessage::Uci => {
                println!("id name smtbot");
                println!("id author smt");
                println!("{}", UciMessage::UciOk);
            }
            UciMessage::IsReady => println!("{}", UciMessage::ReadyOk),
            UciMessage::UciNewGame => {
                state = State::init();
                println!("{}", UciMessage::UciOk);
            }
            UciMessage::Position {
                startpos: true,
                moves,
                ..
            } => {
                for uci_move in moves {
                    let from = sq(uci_move.from);
                    let to = sq(uci_move.to);
                    uci_move.promotion
                    let shakmaty_move: Move = Move::Normal { role: 0, from, capture: None, to, promotion: None };
                    state.pos.play_unchecked(&shakmaty_move);
                }
            }
            UciMessage::Go { .. } => {
                println!("{}", UciMessage::UciOk);
            }
            UciMessage::Stop | UciMessage::Quit => {
                println!("{}", UciMessage::UciOk);
                exit(0);
            }
            _ => writeln!(log_file, "Unexpected message: {:?}", msg).unwrap(),
        }
        println!("{}", UciMessage::UciOk);
    }
}

fn piece_value(role: Role) -> u8 {
    match role {
        Role::Pawn => 1,
        Role::Knight => 3,
        Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}
