use rand;
use rand::prelude::SliceRandom;
use shakmaty::uci::Uci;
use shakmaty::{Chess, Color, Move, Position, Role};
use std::io::Write;
use std::io::{self, BufRead};
use std::process::exit;
use std::str::FromStr;
use vampirc_uci::{parse_one, UciMessage, UciSquare};

struct State {
    pos: Chess,
    white_sum: u8,
    black_sum: u8,
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
    let mut rng = rand::thread_rng();

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = dbg!(parse_one(&line.unwrap()));
        write!(log_file, "{:?}\n", msg).unwrap();
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
                // the client always sends all of the moves, so the only one we're interested in, is the last one
                if moves.len() == 0 {
                    continue;
                }
                let last_move = moves.last().unwrap();
                let uci = Uci::from_str(&format!("{}", last_move)).unwrap();
                let mov = uci.to_move(&mut state.pos).unwrap();
                state.pos.play_unchecked(&mov);
                println!("{}", UciMessage::UciOk);

                // this is where we're gonna make a move
                let move_list = state.pos.legal_moves();
                let our_reply = move_list.choose(&mut rng).unwrap();
                state.pos.play_unchecked(our_reply);
                let uci = our_reply.to_uci(state.pos.castles().mode());
                println!("bestmove {}", uci);
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
