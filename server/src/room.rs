//! One game room: its seats, the game in progress and the agents playing in it.
//!
//! Each room runs on its own thread, which owns the game state and the agents (agents aren't
//! `Send`, and searching agents block for a while). Connections send it [`Command`]s; it sends
//! every connection its own view of the game as JSON.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use rainbow::agents::Agent;
use rainbow::constants::{MAX_RANK, MIN_RANK};
use rainbow::{Combo, ComboKind, GameState};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde_json::{Value, json};
use tokio::sync::mpsc::UnboundedSender;

use crate::presets::Preset;

/// How long an agent waits before playing, so people can follow the game.
const AGENT_DELAY: Duration = Duration::from_millis(700);
/// The pause after a trick completes, so everyone can see how it ended.
const TRICK_PAUSE: Duration = Duration::from_millis(2000);
/// Multiplies both delays (`--pace`); 0 plays agents' moves at once, for testing.
pub static PACE: OnceLock<f32> = OnceLock::new();

fn paced(delay: Duration) -> Duration {
    delay.mul_f32(*PACE.get().unwrap_or(&1.0))
}
/// A room with nobody connected closes after this long.
const ABANDONED_AFTER: Duration = Duration::from_secs(15 * 60);
pub const MAX_NAME_LEN: usize = 20;
const MAX_CHAT_LEN: usize = 200;
/// Minimum time between one connection's chat messages.
const CHAT_INTERVAL: Duration = Duration::from_millis(700);

/// A connection's outgoing message channel.
#[derive(Debug, Clone)]
pub struct Conn {
    pub id: u64,
    pub tx: UnboundedSender<String>,
}

impl Conn {
    fn send(&self, msg: &Value) {
        // A closed channel means the connection is gone; its disconnect command follows.
        let _ = self.tx.send(msg.to_string());
    }
}

pub fn error(msg: &str) -> String {
    json!({ "t": "error", "msg": msg }).to_string()
}

#[derive(Debug)]
pub enum Command {
    /// Takes a seat (or the seat held by `token`, after a reconnect), or watches if the room is
    /// full.
    Join {
        name: String,
        token: Option<String>,
        conn: Conn,
    },
    Disconnect {
        conn: u64,
    },
    /// Host only: starts a game, or the next one once a game is over.
    Start {
        conn: u64,
    },
    Play {
        conn: u64,
        action: usize,
    },
    /// Asks for a suggested move. Only offered when no other person is seated, so hints can't
    /// help anyone against other people.
    Hint {
        conn: u64,
    },
    /// A chat message to everyone in the room.
    Chat {
        conn: u64,
        text: String,
    },
    /// Host only: sets a seat to an agent preset, or opens it for a person (`None`). A seat
    /// held by a connected person can't be changed.
    SetSeat {
        conn: u64,
        seat: usize,
        agent: Option<String>,
    },
}

enum Seat {
    Open,
    Human {
        name: String,
        token: String,
        conn: Option<Conn>,
    },
    Agent {
        preset: String,
        label: String,
    },
}

struct Game {
    state: GameState,
    agents: Vec<Option<Box<dyn Agent>>>,
    /// The last completed trick: its plays and the points each seat took from it.
    last_trick: Option<Value>,
    /// Every completed trick this game, oldest first.
    log: Vec<Value>,
    /// When the agent to move may play.
    agent_at: Instant,
}

pub struct Room {
    code: String,
    seats: Vec<Seat>,
    spectators: Vec<Conn>,
    /// Token of the player who created the room.
    host: Option<String>,
    game: Option<Game>,
    games_played: u32,
    rng: SmallRng,
    presets: Arc<Vec<Preset>>,
    rooms: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<Command>>>>,
    /// When each connection last chatted, for rate limiting.
    last_chat: HashMap<u64, Instant>,
    /// When the last person disconnected, while nobody is connected.
    empty_since: Option<Instant>,
}

/// The cards of a combo, lowest first.
fn cards(combo: Combo) -> Vec<u8> {
    (MIN_RANK..=MAX_RANK)
        .flat_map(|r| std::iter::repeat_n(r, combo.count_of(r).into()))
        .collect()
}

fn play_json(seat: usize, combo: Combo) -> Value {
    json!({ "seat": seat, "cards": cards(combo), "label": combo.to_string() })
}

impl Room {
    /// `seats` holds a preset id per seat, or `None` for a seat open to people.
    pub fn new(
        code: String,
        seats: Vec<Option<String>>,
        presets: Arc<Vec<Preset>>,
        rooms: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<Command>>>>,
    ) -> Self {
        let mut room = Self {
            code,
            seats: Vec::new(),
            spectators: Vec::new(),
            host: None,
            game: None,
            games_played: 0,
            rng: SmallRng::from_rng(&mut rand::rng()),
            presets,
            rooms,
            last_chat: HashMap::new(),
            empty_since: Some(Instant::now()),
        };
        room.seats = seats
            .into_iter()
            .map(|s| s.and_then(|id| room.agent_seat(&id)).unwrap_or(Seat::Open))
            .collect();
        room
    }

    fn agent_seat(&self, id: &str) -> Option<Seat> {
        self.presets
            .iter()
            .find(|p| p.id == id)
            .map(|p| Seat::Agent {
                preset: p.id.clone(),
                label: p.label.clone(),
            })
    }

    pub fn run(mut self, rx: Receiver<Command>) {
        loop {
            let now = Instant::now();
            let wait = self.agent_due().map_or(Duration::from_secs(30), |at| {
                at.saturating_duration_since(now)
            });
            match rx.recv_timeout(wait) {
                Ok(cmd) => self.handle(cmd),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            if self.agent_due().is_some_and(|at| at <= Instant::now()) {
                self.agent_move();
            }
            if self
                .empty_since
                .is_some_and(|t| t.elapsed() >= ABANDONED_AFTER)
            {
                break;
            }
        }
        self.rooms.lock().expect("lock").remove(&self.code);
    }

    fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Join { name, token, conn } => self.join(name, token, conn),
            Command::Disconnect { conn } => {
                for seat in &mut self.seats {
                    if let Seat::Human { conn: c, .. } = seat
                        && c.as_ref().is_some_and(|c| c.id == conn)
                    {
                        *c = None;
                    }
                }
                self.spectators.retain(|c| c.id != conn);
                self.last_chat.remove(&conn);
                // A seat left before the first game opens up again.
                if self.game.is_none() && self.games_played == 0 {
                    for seat in &mut self.seats {
                        if matches!(seat, Seat::Human { conn: None, .. }) {
                            *seat = Seat::Open;
                        }
                    }
                }
                if self.connections().next().is_none() {
                    self.empty_since = Some(Instant::now());
                }
                self.broadcast();
            }
            Command::Start { conn } => {
                if !self.is_host(conn) {
                    return self.reply(conn, "only the host can start the game");
                }
                if self.game.as_ref().is_some_and(|g| !g.state.is_over()) {
                    return self.reply(conn, "a game is already in progress");
                }
                if self.seats.iter().any(|s| matches!(s, Seat::Open)) {
                    return self.reply(conn, "fill or close every open seat first");
                }
                self.start();
            }
            Command::Play { conn, action } => self.human_move(conn, action),
            Command::Hint { conn } => self.hint(conn),
            Command::Chat { conn, text } => self.chat(conn, &text),
            Command::SetSeat { conn, seat, agent } => {
                if !self.is_host(conn) {
                    return self.reply(conn, "only the host can change seats");
                }
                let Some(current) = self.seats.get(seat) else {
                    return self.reply(conn, "no such seat");
                };
                if matches!(current, Seat::Human { conn: Some(_), .. }) {
                    return self.reply(conn, "that seat belongs to a connected player");
                }
                let new = match agent {
                    Some(id) => match self.agent_seat(&id) {
                        Some(s) => s,
                        None => return self.reply(conn, "unknown agent"),
                    },
                    None => Seat::Open,
                };
                // During a game a seat can only be handed to an agent, never left empty.
                if self.game.as_ref().is_some_and(|g| !g.state.is_over()) {
                    let Seat::Agent { preset, .. } = &new else {
                        return self.reply(conn, "a seat can't be opened during a game");
                    };
                    let agent = self
                        .presets
                        .iter()
                        .find(|p| &p.id == preset)
                        .and_then(|p| p.create(self.rng.random()));
                    let game = self.game.as_mut().expect("checked above");
                    game.agents[seat] = agent;
                    game.agent_at = Instant::now() + paced(AGENT_DELAY);
                }
                self.seats[seat] = new;
                self.broadcast();
            }
        }
    }

    fn join(&mut self, name: String, token: Option<String>, conn: Conn) {
        self.empty_since = None;
        let name: String = name.trim().chars().take(MAX_NAME_LEN).collect();
        let name = if name.is_empty() {
            "Player".into()
        } else {
            name
        };
        // Reclaim a seat after a reconnect.
        let reclaimed = token.as_ref().and_then(|t| {
            self.seats
                .iter()
                .position(|s| matches!(s, Seat::Human { token, .. } if token == t))
        });
        let seat = reclaimed.or_else(|| self.seats.iter().position(|s| matches!(s, Seat::Open)));
        let Some(seat) = seat else {
            conn.send(&json!({ "t": "joined", "code": self.code, "seat": null, "host": false }));
            self.spectators.push(conn);
            return self.broadcast();
        };
        let token = match &self.seats[seat] {
            Seat::Human { token, .. } => token.clone(),
            _ => format!("{:016x}", self.rng.random::<u64>()),
        };
        if let Seat::Human {
            conn: Some(old), ..
        } = &self.seats[seat]
        {
            old.send(&json!({ "t": "error", "msg": "this seat was taken over by another window" }));
        }
        self.host.get_or_insert_with(|| token.clone());
        conn.send(&json!({
            "t": "joined",
            "code": self.code,
            "seat": seat,
            "token": token,
            "host": self.host.as_ref() == Some(&token),
        }));
        self.seats[seat] = Seat::Human {
            name,
            token,
            conn: Some(conn),
        };
        self.broadcast();
    }

    fn connections(&self) -> impl Iterator<Item = (Option<usize>, &Conn)> {
        self.seats
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Seat::Human { conn: Some(c), .. } => Some((Some(i), c)),
                _ => None,
            })
            .chain(self.spectators.iter().map(|c| (None, c)))
    }

    fn seat_of(&self, conn: u64) -> Option<usize> {
        self.seats
            .iter()
            .position(|s| matches!(s, Seat::Human { conn: Some(c), .. } if c.id == conn))
    }

    fn is_host(&self, conn: u64) -> bool {
        self.seat_of(conn).is_some_and(|i| {
            matches!(&self.seats[i], Seat::Human { token, .. } if Some(token) == self.host.as_ref())
        })
    }

    fn reply(&self, conn: u64, msg: &str) {
        if let Some((_, c)) = self.connections().find(|(_, c)| c.id == conn) {
            let _ = c.tx.send(error(msg));
        }
    }

    fn start(&mut self) {
        let n = self.seats.len();
        let leader = self.rng.random_range(0..n);
        let state = GameState::new(n, leader, &mut self.rng);
        let agents = self
            .seats
            .iter()
            .map(|s| match s {
                Seat::Agent { preset, .. } => self
                    .presets
                    .iter()
                    .find(|p| &p.id == preset)
                    .and_then(|p| p.create(self.rng.random())),
                _ => None,
            })
            .collect();
        self.game = Some(Game {
            state,
            agents,
            last_trick: None,
            log: Vec::new(),
            agent_at: Instant::now() + paced(AGENT_DELAY),
        });
        self.games_played += 1;
        self.broadcast();
    }

    fn agent_due(&self) -> Option<Instant> {
        let game = self.game.as_ref()?;
        if game.state.is_over() {
            return None;
        }
        game.agents[game.state.current_player()]
            .as_ref()
            .map(|_| game.agent_at)
    }

    fn agent_move(&mut self) {
        let Some(game) = &mut self.game else { return };
        let seat = game.state.current_player();
        let Some(agent) = &mut game.agents[seat] else {
            return;
        };
        let legal = game.state.legal_actions();
        let combo = agent.act(&game.state.observe(seat), &legal);
        self.apply(combo);
    }

    fn chat(&mut self, conn: u64, text: &str) {
        let text: String = text.trim().chars().take(MAX_CHAT_LEN).collect();
        if text.is_empty() {
            return;
        }
        let now = Instant::now();
        if self
            .last_chat
            .get(&conn)
            .is_some_and(|t| now.duration_since(*t) < CHAT_INTERVAL)
        {
            return self.reply(conn, "you're sending messages too quickly");
        }
        self.last_chat.insert(conn, now);
        let from = match self.seat_of(conn).map(|i| &self.seats[i]) {
            Some(Seat::Human { name, .. }) => name.clone(),
            _ => "Spectator".into(),
        };
        let msg = json!({ "t": "chat", "from": from, "seat": self.seat_of(conn), "text": text });
        for (_, c) in self.connections() {
            c.send(&msg);
        }
    }

    /// Whether hints are allowed: no other person is seated.
    fn hints_allowed(&self) -> bool {
        self.seats
            .iter()
            .filter(|s| matches!(s, Seat::Human { .. }))
            .count()
            <= 1
    }

    fn hint(&mut self, conn: u64) {
        if !self.hints_allowed() {
            return self.reply(conn, "hints are only available in games against agents");
        }
        let (Some(seat), Some(game)) = (self.seat_of(conn), &self.game) else {
            return self.reply(conn, "the game hasn't started");
        };
        if game.state.is_over() || game.state.current_player() != seat {
            return self.reply(conn, "it isn't your turn");
        }
        // The strongest fast agent available gives the hint.
        let Some(mut adviser) = ["dmc", "counter"]
            .iter()
            .find_map(|id| self.presets.iter().find(|p| p.id == *id))
            .and_then(|p| p.create(self.rng.random()))
        else {
            return self.reply(conn, "no agent is available to give hints");
        };
        let combo = adviser.act(&game.state.observe(seat), &game.state.legal_actions());
        if let Some((_, c)) = self.connections().find(|(_, c)| c.id == conn) {
            c.send(&json!({ "t": "hint", "action": combo.action_index(), "cards": cards(combo) }));
        }
    }

    fn human_move(&mut self, conn: u64, action: usize) {
        let Some(seat) = self.seat_of(conn) else {
            return self.reply(conn, "you aren't seated in this room");
        };
        let Some(game) = &self.game else {
            return self.reply(conn, "the game hasn't started");
        };
        if game.state.is_over() || game.state.current_player() != seat {
            return self.reply(conn, "it isn't your turn");
        }
        if game.agents[seat].is_some() {
            return self.reply(conn, "an agent is playing this seat");
        }
        match Combo::from_action_index(action).filter(|&c| game.state.is_legal(c)) {
            Some(combo) => self.apply(combo),
            None => self.reply(conn, "that play isn't legal"),
        }
    }

    fn apply(&mut self, combo: Combo) {
        let Some(game) = &mut self.game else { return };
        let seat = game.state.current_player();
        let before: Vec<u8> = game.state.scores().to_vec();
        let mut plays: Vec<Value> = game
            .state
            .trick()
            .iter()
            .map(|p| play_json(p.seat.into(), p.combo))
            .collect();
        plays.push(play_json(seat, combo));
        game.state
            .apply(combo)
            .expect("only legal plays are applied");
        let completed = game.state.trick().is_empty() || game.state.is_over();
        let mut delay = AGENT_DELAY;
        if completed {
            let taken: Vec<u8> = game
                .state
                .scores()
                .iter()
                .zip(&before)
                .map(|(after, before)| after - before)
                .collect();
            let summary = json!({ "plays": plays, "taken": taken });
            game.log.push(summary.clone());
            game.last_trick = Some(summary);
            delay = TRICK_PAUSE;
        }
        game.agent_at = Instant::now() + paced(delay);
        self.broadcast();
    }

    fn broadcast(&self) {
        let room = self.room_json();
        for (seat, conn) in self.connections() {
            conn.send(&room);
            if let Some(view) = self.game_json(seat) {
                conn.send(&view);
            }
        }
    }

    fn room_json(&self) -> Value {
        let seats: Vec<Value> = self
            .seats
            .iter()
            .map(|s| match s {
                Seat::Open => json!({ "kind": "open" }),
                Seat::Human { name, token, conn } => json!({
                    "kind": "human",
                    "name": name,
                    "connected": conn.is_some(),
                    "host": Some(token) == self.host.as_ref(),
                }),
                Seat::Agent { preset, label } => {
                    json!({ "kind": "agent", "name": label, "preset": preset })
                }
            })
            .collect();
        // Number agents that share a name: "DMC 1", "DMC 2".
        let mut seats = seats;
        for i in 0..seats.len() {
            if seats[i]["kind"] != "agent" {
                continue;
            }
            let same = |s: &Value| s["kind"] == "agent" && s["preset"] == seats[i]["preset"];
            if seats.iter().filter(|s| same(s)).count() > 1 {
                let k = seats[..i].iter().filter(|s| same(s)).count() + 1;
                let name = format!("{} {k}", seats[i]["name"].as_str().unwrap_or("Agent"));
                seats[i]["name"] = Value::from(name);
            }
        }
        let phase = match &self.game {
            None => "lobby",
            Some(g) if g.state.is_over() => "over",
            Some(_) => "playing",
        };
        json!({
            "t": "room",
            "code": self.code,
            "seats": seats,
            "phase": phase,
            "games_played": self.games_played,
            "hints": self.hints_allowed(),
            "spectators": self.spectators.len(),
        })
    }

    /// The game as `viewer` sees it: everything public, plus their own hand and legal plays.
    fn game_json(&self, viewer: Option<usize>) -> Option<Value> {
        let game = self.game.as_ref()?;
        let s = &game.state;
        let over = s.is_over();
        let players: Vec<Value> = (0..s.num_players())
            .map(|i| {
                let obs = s.observe(i);
                let p = obs.players()[i];
                json!({
                    "hand_size": p.hand_size,
                    "reserve_size": p.reserve_size,
                    "points": p.points,
                })
            })
            .collect();
        let hand = viewer.map(|v| s.hand(v).cards().collect::<Vec<u8>>());
        let my_turn = viewer.is_some_and(|v| !over && s.current_player() == v);
        // The play leading the trick: the strongest, ties going to the earlier play.
        let best = s.trick().iter().map(|p| p.combo.strength()).max();
        let leader = best.and_then(|b| s.trick().iter().position(|p| p.combo.strength() == b));
        let legal: Vec<Value> = if my_turn {
            s.legal_actions()
                .iter()
                .map(|&c| {
                    let kind = match c.kind() {
                        ComboKind::Single { .. } => "single",
                        ComboKind::Set { .. } => "set",
                        ComboKind::Run { .. } => "run",
                    };
                    json!({
                        "action": c.action_index(),
                        "cards": cards(c),
                        "label": c.to_string(),
                        "kind": kind,
                        "beats": best.is_none_or(|b| c.strength() > b),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let trick: Vec<Value> = s
            .trick()
            .iter()
            .map(|p| play_json(p.seat.into(), p.combo))
            .collect();
        // Cards the viewer hasn't seen: other hands, reserves and set-aside cards.
        let unseen = viewer.map(|v| s.observe(v).unknown().counts());
        let trick_type = format!("{:?}", s.trick_type()).to_lowercase();
        Some(json!({
            "t": "state",
            "seat": viewer,
            "current": if over { None } else { Some(s.current_player()) },
            "players": players,
            "piles": s.piles(),
            "trick": trick,
            "trick_leader": leader,
            "unseen": unseen,
            "log": game.log,
            "trick_type": trick_type,
            "hand": hand,
            "legal": legal,
            "last_trick": game.last_trick,
            "tricks_played": s.tricks_played(),
            "over": over,
            "winners": if over { Some(s.winners().to_vec()) } else { None },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    #[test]
    fn a_person_plays_a_whole_game_against_an_agent() {
        PACE.get_or_init(|| 0.0);
        let presets = Arc::new(presets::available(std::path::Path::new("no-models")));
        let rooms = Arc::default();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            Room::new(
                "TEST".into(),
                vec![None, Some("random".into())],
                presets,
                rooms,
            )
            .run(rx)
        });
        let (out, mut inbox) = tokio::sync::mpsc::unbounded_channel();
        let conn = Conn { id: 1, tx: out };
        tx.send(Command::Join {
            name: "Ann".into(),
            token: None,
            conn,
        })
        .unwrap();
        tx.send(Command::Start { conn: 1 }).unwrap();
        let mut moves = 0;
        loop {
            let msg: Value = serde_json::from_str(&inbox.blocking_recv().unwrap()).unwrap();
            assert_ne!(msg["t"], "error", "{msg}");
            if msg["t"] != "state" {
                continue;
            }
            if msg["over"] == true {
                assert!(!msg["winners"].as_array().unwrap().is_empty());
                break;
            }
            if let Some(play) = msg["legal"].as_array().and_then(|l| l.first()) {
                let action = play["action"].as_u64().unwrap() as usize;
                tx.send(Command::Play { conn: 1, action }).unwrap();
                moves += 1;
            }
        }
        assert!(moves > 0);
    }

    #[test]
    fn only_the_host_starts_and_open_seats_block_the_start() {
        PACE.get_or_init(|| 0.0);
        let presets = Arc::new(presets::available(std::path::Path::new("no-models")));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            Room::new(
                "TST2".into(),
                vec![None, None, None],
                presets,
                Arc::default(),
            )
            .run(rx)
        });
        let mut inboxes = Vec::new();
        for id in 1..=2 {
            let (out, inbox) = tokio::sync::mpsc::unbounded_channel();
            inboxes.push(inbox);
            let conn = Conn { id, tx: out };
            tx.send(Command::Join {
                name: format!("P{id}"),
                token: None,
                conn,
            })
            .unwrap();
        }
        let next_error = |inbox: &mut tokio::sync::mpsc::UnboundedReceiver<String>| loop {
            let msg: Value = serde_json::from_str(&inbox.blocking_recv().unwrap()).unwrap();
            if msg["t"] == "error" {
                return msg["msg"].as_str().unwrap().to_string();
            }
        };
        tx.send(Command::Start { conn: 2 }).unwrap();
        assert_eq!(
            next_error(&mut inboxes[1]),
            "only the host can start the game"
        );
        tx.send(Command::Start { conn: 1 }).unwrap();
        assert_eq!(
            next_error(&mut inboxes[0]),
            "fill or close every open seat first"
        );
    }
}
