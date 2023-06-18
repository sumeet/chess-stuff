use rand;
use rand::prelude::SliceRandom;
use shakmaty::uci::Uci;
use shakmaty::{Chess, Color, Move, Outcome, Position, Role};
use std::fs::File;
use std::io::{self, BufRead};
use std::io::{BufReader, Write};
use std::num::{NonZeroU32, NonZeroU8};
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

use pgn_reader::{BufferedReader, SanPlus, Visitor};
use shakmaty::san::San;

struct GameCollector {
    current_pos: Chess,
    moves: Vec<Move>,
    num_games_collected: usize,
    output: File,
}

impl GameCollector {
    fn new() -> Self {
        Self {
            moves: vec![],
            current_pos: Chess::default(),
            num_games_collected: 0,
            output: File::create("output.chess").unwrap(),
        }
    }
}

enum ParseResult {
    KeepGoing,
    Done,
}

impl Visitor for GameCollector {
    type Result = ParseResult;

    fn san(&mut self, san: SanPlus) {
        let m = San::from_str(&format!("{}", san))
            .unwrap()
            .to_move(&self.current_pos)
            .unwrap();
        self.current_pos = self.current_pos.clone().play(&m).unwrap();
        self.moves.push(m);
    }

    fn end_game(&mut self) -> Self::Result {
        use ParseResult::*;

        let mut end_pos = Chess::default();
        std::mem::swap(&mut end_pos, &mut self.current_pos);

        let mut end_moves = vec![];
        std::mem::swap(&mut end_moves, &mut self.moves);

        // if end_pos.fullmoves() > unsafe { NonZeroU32::new_unchecked(50) } {
        //     return KeepGoing;
        // }

        let outcome = end_pos.outcome();
        // for now also filter out draws because we only want to choose
        // moves from the winner
        if let None | Some(Outcome::Draw) = outcome {
            return KeepGoing;
        }
        let outcome = outcome.unwrap();

        let (last_move, moves) = end_moves.split_last().unwrap();
        for m in moves {
            write!(self.output, "{},", Uci::from_standard(&m)).unwrap();
        }
        write!(self.output, "{}\n", Uci::from_standard(&last_move)).unwrap();

        match outcome {
            Outcome::Decisive { winner } => match winner {
                Color::Black => write!(self.output, "B\n").unwrap(),
                Color::White => write!(self.output, "W\n").unwrap(),
            },
            Outcome::Draw => write!(self.output, "D\n").unwrap(),
        }
        self.num_games_collected += 1;
        KeepGoing
        // if self.num_games_collected > 200_000 {
        //     Done
        // } else {
        //     KeepGoing
        // }
    }
}

fn main() {
    let f = File::open("./download/elite.pgn.gz").unwrap();
    let reader = BufReader::new(f);
    let reader = flate2::bufread::GzDecoder::new(reader);
    let mut pgn_reader = BufferedReader::new(reader);
    let mut game_collector = GameCollector::new();
    while let Some(parse_result) = pgn_reader.read_game(&mut game_collector).unwrap() {
        match parse_result {
            ParseResult::KeepGoing => {}
            ParseResult::Done => break,
        }
    }
}

fn main2() {
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
