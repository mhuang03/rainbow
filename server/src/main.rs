//! A web server for playing Rainbow in the browser: alone against the agents, or online with
//! other people in shared rooms, with any seats left over filled by agents.
//!
//! ```sh
//! cargo run --release -p rainbow-server -- --port 8080 --models models
//! ```
//!
//! Then open http://localhost:8080. Other people join a room through its link or four-letter
//! code, so for online play the port must be reachable from their machines.

mod presets;
mod room;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use rainbow::constants::{MAX_PLAYERS, MIN_PLAYERS};
use rand::RngExt;
use serde::Deserialize;

use presets::Preset;
use room::{Command, Conn, Room, error};

const MAX_ROOMS: usize = 500;
/// Room codes avoid letters that are easy to confuse (I, L, O).
const CODE_LETTERS: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ";

struct App {
    rooms: Arc<Mutex<HashMap<String, Sender<Command>>>>,
    presets: Arc<Vec<Preset>>,
    next_conn: AtomicU64,
}

#[derive(Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
enum ClientMsg {
    /// Creates a room and takes its first seat. `seats` lists the other seats: an agent preset
    /// id, or "open" for a person.
    Create {
        name: String,
        seats: Vec<String>,
        /// Every seat in `seats` goes to the computer and the creator watches.
        #[serde(default)]
        watch: bool,
    },
    Join {
        code: String,
        name: String,
        token: Option<String>,
    },
    Leave,
    Start,
    Play {
        action: usize,
    },
    Hint,
    Speed {
        fast: bool,
    },
    Chat {
        text: String,
    },
    Seat {
        seat: usize,
        agent: Option<String>,
    },
}

fn usage() -> ! {
    eprintln!("usage: rainbow-server [--host ADDR] [--port PORT] [--models DIR] [--pace FACTOR]");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let mut host = "0.0.0.0".to_string();
    let mut port = 8080u16;
    let mut models = PathBuf::from("models");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--host" => host = value(),
            "--port" => port = value().parse().unwrap_or_else(|_| usage()),
            "--models" => models = value().into(),
            "--pace" => {
                let pace = value().parse().unwrap_or_else(|_| usage());
                room::PACE.set(pace).expect("set once");
            }
            _ => usage(),
        }
    }
    let presets = presets::available(&models);
    eprintln!(
        "agents: {}",
        presets
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let app = Arc::new(App {
        rooms: Arc::default(),
        presets: Arc::new(presets),
        next_conn: AtomicU64::new(1),
    });
    let router = Router::new()
        .route(
            "/",
            get(|| async { static_file("text/html", include_str!("../static/index.html")) }),
        )
        .route(
            "/app.js",
            get(|| async { static_file("text/javascript", include_str!("../static/app.js")) }),
        )
        .route(
            "/style.css",
            get(|| async { static_file("text/css", include_str!("../static/style.css")) }),
        )
        .route("/api/agents", get(agents_list))
        .route("/ws", get(ws))
        .with_state(app);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("can't listen on {addr}: {e}"));
    eprintln!("listening on http://{addr}");
    axum::serve(listener, router).await.expect("server error");
}

fn static_file(mime: &'static str, body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, mime)], body).into_response()
}

async fn agents_list(State(app): State<Arc<App>>) -> Response {
    axum::Json(app.presets.as_ref().clone()).into_response()
}

/// Largest message a browser may send; real ones (a play, a chat line) are well under 1 KiB.
const MAX_MESSAGE: usize = 16 * 1024;

async fn ws(upgrade: WebSocketUpgrade, State(app): State<Arc<App>>) -> Response {
    upgrade
        .max_message_size(MAX_MESSAGE)
        .max_frame_size(MAX_MESSAGE)
        .on_upgrade(move |socket| connection(socket, app))
}

fn new_code(rooms: &HashMap<String, Sender<Command>>) -> String {
    let mut rng = rand::rng();
    loop {
        let code: String = (0..4)
            .map(|_| CODE_LETTERS[rng.random_range(0..CODE_LETTERS.len())] as char)
            .collect();
        if !rooms.contains_key(&code) {
            return code;
        }
    }
}

/// Serves one browser connection: forwards its commands to its room and the room's messages
/// back to it.
async fn connection(mut socket: WebSocket, app: Arc<App>) {
    let id = app.next_conn.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let conn = Conn { id, tx };
    let mut room: Option<Sender<Command>> = None;
    loop {
        tokio::select! {
            incoming = socket.recv() => {
                let Some(Ok(msg)) = incoming else { break };
                let Message::Text(text) = msg else {
                    if matches!(msg, Message::Close(_)) { break }
                    continue;
                };
                match serde_json::from_str::<ClientMsg>(&text) {
                    Ok(msg) => handle(msg, &app, &conn, &mut room),
                    Err(e) => { let _ = conn.tx.send(error(&format!("bad message: {e}"))); }
                }
            }
            outgoing = rx.recv() => {
                let Some(text) = outgoing else { break };
                if socket.send(Message::Text(text.into())).await.is_err() { break }
            }
        }
    }
    if let Some(room) = room {
        let _ = room.send(Command::Disconnect { conn: id });
    }
}

fn handle(msg: ClientMsg, app: &App, conn: &Conn, room: &mut Option<Sender<Command>>) {
    let send_error = |m: &str| {
        let _ = conn.tx.send(error(m));
    };
    let command = match msg {
        ClientMsg::Create { name, seats, watch } => {
            let players = seats.len() + usize::from(!watch);
            if !(MIN_PLAYERS..=MAX_PLAYERS).contains(&players) {
                return send_error("a game has 2 to 6 players");
            }
            let mut rooms = app.rooms.lock().expect("lock");
            if rooms.len() >= MAX_ROOMS {
                return send_error("the server is busy right now; try again in a few minutes");
            }
            let code = new_code(&rooms);
            if watch && seats.iter().any(|s| s == "open") {
                return send_error("a watched game has only computer players");
            }
            let seats = (!watch)
                .then_some(None)
                .into_iter()
                .chain(seats.into_iter().map(|s| (s != "open").then_some(s)))
                .collect();
            let (tx, rx) = std::sync::mpsc::channel();
            let (c, presets, registry) = (
                code.clone(),
                Arc::clone(&app.presets),
                Arc::clone(&app.rooms),
            );
            // The room owns agents, which aren't `Send`, so it is built on its own thread.
            std::thread::spawn(move || {
                let room = Room::new(c, seats, presets, registry);
                if watch { room.watching() } else { room }.run(rx)
            });
            rooms.insert(code, tx.clone());
            leave(room, conn);
            *room = Some(tx);
            Command::Join {
                name,
                token: None,
                conn: conn.clone(),
            }
        }
        ClientMsg::Join { code, name, token } => {
            let code = code.trim().to_uppercase();
            let Some(tx) = app.rooms.lock().expect("lock").get(&code).cloned() else {
                return send_error("that room doesn't exist or has closed");
            };
            leave(room, conn);
            *room = Some(tx);
            Command::Join {
                name,
                token,
                conn: conn.clone(),
            }
        }
        ClientMsg::Leave => return leave(room, conn),
        ClientMsg::Start => Command::Start { conn: conn.id },
        ClientMsg::Hint => Command::Hint { conn: conn.id },
        ClientMsg::Speed { fast } => Command::Speed {
            conn: conn.id,
            fast,
        },
        ClientMsg::Chat { text } => Command::Chat {
            conn: conn.id,
            text,
        },
        ClientMsg::Play { action } => Command::Play {
            conn: conn.id,
            action,
        },
        ClientMsg::Seat { seat, agent } => Command::SetSeat {
            conn: conn.id,
            seat,
            agent,
        },
    };
    match room {
        Some(tx) => {
            if tx.send(command).is_err() {
                *room = None;
                send_error("the room has closed");
            }
        }
        None => send_error("join a room first"),
    }
}

/// Leaves the current room before joining another.
fn leave(room: &mut Option<Sender<Command>>, conn: &Conn) {
    if let Some(old) = room.take() {
        let _ = old.send(Command::Disconnect { conn: conn.id });
    }
}
