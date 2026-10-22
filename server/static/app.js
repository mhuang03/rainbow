"use strict";

// ---------- state ----------
const me = { code: null, seat: null, token: null, host: false };
let ws = null;
let room = null;      // latest "room" message
let game = null;      // latest "state" message
let presets = [];
let selection = [];   // indices into game.hand
let autoStart = false;

const $ = (id) => document.getElementById(id);
const TRICK_TYPES = { open: "any combo", set: "singles or sets", run: "singles or runs" };
const ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th"];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function card(rank, small) {
  return el("div", `card r${rank}${small ? " small" : ""}`, String(rank));
}

function cardRow(ranks, small) {
  const row = el("div", "cards");
  ranks.forEach((r) => row.append(card(r, small)));
  return row;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

function playerName() {
  const name = $("name").value.trim() || "Player";
  localStorage.setItem("rainbow.name", name);
  return name;
}

function seatName(seat) {
  const s = room && room.seats[seat];
  if (!s) return `Seat ${seat + 1}`;
  if (s.kind === "open") return "Open seat";
  return s.name;
}

// ---------- connection ----------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    $("status").textContent = "";
    if (me.code) send({ t: "join", code: me.code, name: playerName(), token: me.token });
  };
  ws.onmessage = (ev) => receive(JSON.parse(ev.data));
  ws.onclose = () => {
    $("status").textContent = "Reconnecting…";
    setTimeout(connect, 1500);
  };
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  else toast("Not connected yet, try again in a moment");
}

function receive(msg) {
  switch (msg.t) {
    case "joined":
      me.code = msg.code;
      me.seat = msg.seat;
      me.token = msg.token || null;
      me.host = msg.host;
      if (me.token) localStorage.setItem(`rainbow.token.${me.code}`, me.token);
      history.replaceState(null, "", `/?room=${me.code}`);
      break;
    case "room":
      room = msg;
      if (room.phase === "lobby") game = null;
      if (autoStart && me.host && room.phase === "lobby" && !room.seats.some((s) => s.kind === "open")) {
        autoStart = false;
        send({ t: "start" });
      }
      break;
    case "state": {
      const oldHand = game && game.hand ? game.hand.join() : "";
      announceChanges(game && game.tricks_played <= msg.tricks_played ? game : null, msg);
      game = msg;
      if (!game.hand || game.hand.join() !== oldHand) selection = [];
      break;
    }
    case "error":
      toast(msg.msg);
      if (msg.msg === "no room with that code" || msg.msg === "the room has closed") leave();
      break;
  }
  render();
}

function leave() {
  me.code = null;
  me.seat = null;
  me.token = null;
  me.host = false;
  room = null;
  game = null;
  history.replaceState(null, "", "/");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ t: "leave" }));
  render();
}

// ---------- home ----------
function presetOptions(select, withOpen, value) {
  select.innerHTML = "";
  if (withOpen) select.append(new Option("Open (a person joins)", "open"));
  presets.forEach((p) => {
    const o = new Option(p.label, p.id);
    o.title = p.description;
    select.append(o);
  });
  if (value) select.value = value;
}

function seatPickers(container, count, withOpen, fallback) {
  const old = [...container.querySelectorAll("select")].map((s) => s.value);
  container.innerHTML = "";
  for (let i = 1; i < count; i++) {
    const label = el("label", null, `Seat ${i + 1}`);
    const select = el("select");
    presetOptions(select, withOpen, old[i - 1] || fallback);
    label.append(select);
    container.append(label);
  }
}

function setupHome() {
  for (const id of ["solo-players", "online-players"]) {
    const s = $(id);
    for (let n = 2; n <= 6; n++) s.append(new Option(`${n} players`, n));
    s.value = 4;
  }
  const defaultAgent = presets.some((p) => p.id === "dmc") ? "dmc" : presets[0] && presets[0].id;
  const refresh = () => {
    seatPickers($("solo-seats"), +$("solo-players").value, false, defaultAgent);
    seatPickers($("online-seats"), +$("online-players").value, true, "open");
  };
  $("solo-players").onchange = refresh;
  $("online-players").onchange = refresh;
  refresh();

  const seatsOf = (id) => [...$(id).querySelectorAll("select")].map((s) => s.value);
  $("solo-start").onclick = () => {
    autoStart = true;
    send({ t: "create", name: playerName(), seats: seatsOf("solo-seats") });
  };
  $("online-create").onclick = () => send({ t: "create", name: playerName(), seats: seatsOf("online-seats") });
  const join = () => {
    const code = $("join-code").value.trim().toUpperCase();
    if (code.length !== 4) return toast("Room codes have four letters");
    me.code = code;
    me.token = localStorage.getItem(`rainbow.token.${code}`);
    send({ t: "join", code, name: playerName(), token: me.token });
  };
  $("join-btn").onclick = join;
  $("join-code").onkeydown = (e) => { if (e.key === "Enter") join(); };
}

// ---------- lobby ----------
function renderLobby() {
  $("lobby-code").textContent = room.code;
  $("share-link").value = `${location.origin}/?room=${room.code}`;
  const list = $("lobby-seats");
  list.innerHTML = "";
  room.seats.forEach((s, i) => {
    const li = el("li");
    const line = el("div", "seat-line");
    const who = el("span");
    if (s.kind === "human") {
      who.append(document.createTextNode(s.name + " "));
      if (i === me.seat) who.append(el("span", "tag you", "you"));
      if (s.host) who.append(el("span", "tag", "host"));
      if (!s.connected) who.append(el("span", "tag off", "away"));
    } else if (s.kind === "agent") {
      who.append(document.createTextNode(s.name + " "), el("span", "tag", "agent"));
    } else {
      who.append(el("span", "hint", "Waiting for a player…"));
    }
    line.append(who);
    const editable = me.host && i !== me.seat && !(s.kind === "human" && s.connected);
    if (editable) {
      const select = el("select");
      presetOptions(select, true, s.kind === "agent" ? s.preset : "open");
      select.onchange = () => send({ t: "seat", seat: i, agent: select.value === "open" ? null : select.value });
      line.append(select);
    }
    li.append(line);
    list.append(li);
  });
  const open = room.seats.filter((s) => s.kind === "open").length;
  $("lobby-start").classList.toggle("hidden", !me.host);
  $("lobby-start").disabled = open > 0;
  $("lobby-hint").textContent = me.host
    ? open > 0 ? `${open} open seat${open > 1 ? "s" : ""}: wait for players or give the seats to agents.` : "Everyone is here."
    : me.seat === null ? "The room is full; you are watching." : "Waiting for the host to start the game.";
}

// ---------- table ----------
const KIND_NAMES = { single: "Singles", set: "Sets", run: "Runs" };
let freshFrom = Infinity; // plays in the current trick from this index on are new: animate them

function who(seat) {
  return seat === game.seat ? "You" : seatName(seat);
}

function cardsText(cards) {
  return cards.join(" ");
}

/** Screen-reader (and log) announcements for what changed between two states. */
function announceChanges(old, next) {
  const live = $("live");
  const said = [];
  const oldLog = old ? old.log.length : 0;
  const oldTrick = old && old.log.length === next.log.length ? old.trick.length : 0;
  if (next.log.length > oldLog) {
    const done = next.log[next.log.length - 1];
    // plays of the finished trick not yet announced
    done.plays.slice(oldLog === next.log.length - 1 && old ? old.trick.length : 0)
      .forEach((pl) => said.push(`${who(pl.seat)} played ${cardsText(pl.cards)}.`));
    const gains = [...new Set(done.plays.map((pl) => pl.seat))]
      .map((seat) => `${who(seat)} +${done.taken[seat]}`);
    said.push(`Trick over: ${gains.join(", ")}.`);
  }
  next.trick.slice(oldTrick).forEach((pl) => said.push(`${who(pl.seat)} played ${cardsText(pl.cards)}.`));
  if (!next.over && next.current === next.seat && (!old || old.current !== next.seat)) said.push("Your turn.");
  if (next.over && (!old || !old.over)) said.push("Game over.");
  if (said.length) live.textContent = said.join(" ");
  freshFrom = oldTrick;
}

function matchingLegal() {
  if (!game || !game.legal.length || !selection.length) return null;
  const ranks = selection.map((i) => game.hand[i]).sort((a, b) => a - b).join();
  return game.legal.find((l) => l.cards.join() === ranks) || null;
}

/** Selects the hand cards of a legal play. */
function selectCombo(l) {
  const used = new Set();
  selection = l.cards.map((r) => {
    const i = game.hand.findIndex((x, j) => x === r && !used.has(j));
    used.add(i);
    return i;
  });
}

function play(action) {
  selection = [];
  send({ t: "play", action });
}

function renderOpponents(spectator, base, n) {
  const opps = $("opponents");
  opps.innerHTML = "";
  for (let k = spectator ? 0 : 1; k < n; k++) {
    const seat = (base + k) % n;
    const p = game.players[seat];
    const s = room.seats[seat] || {};
    const turn = game.current === seat;
    const box = el("div", "opp" + (turn ? " turn" : ""));
    const head = el("div", "who");
    head.append(
      el("span", turn && s.kind === "agent" ? "thinking" : "", seatName(seat)),
      el("span", "kind" + (s.connected === false ? " offline" : ""), s.kind === "agent" ? "agent" : s.connected === false ? "away" : ""),
    );
    const backs = el("div", "backs");
    backs.setAttribute("aria-hidden", "true");
    for (let i = 0; i < p.hand_size; i++) backs.append(el("div", "back"));
    const meta = el("div", "meta");
    meta.append(
      el("span", null, `${p.hand_size} card${p.hand_size === 1 ? "" : "s"}${p.reserve_size ? ` + ${p.reserve_size} reserve` : ""}`),
      el("span", "points", `${p.points} pts`),
    );
    box.append(head, backs, meta);
    if (me.host && s.kind === "human" && !s.connected && !game.over) {
      const select = el("select");
      select.setAttribute("aria-label", `Replace ${seatName(seat)} with an agent`);
      presetOptions(select, false);
      select.prepend(new Option("Replace with an agent…", ""));
      select.value = "";
      select.onchange = () => select.value && send({ t: "seat", seat, agent: select.value });
      box.append(select);
    }
    opps.append(box);
  }
}

function renderMiddle() {
  const piles = $("piles");
  piles.innerHTML = "";
  game.piles.forEach((v, i) => {
    const pile = el("div", "pile");
    pile.setAttribute("aria-label", `${ORDINALS[i]} pile: ${v} points`);
    pile.append(el("b", null, String(v)), el("small", null, ORDINALS[i]));
    piles.append(pile);
  });
  if (!game.piles.length) piles.append(el("span", "hint", "No piles left"));

  // the current trick, or the last finished one until the next trick starts
  const trick = $("trick");
  trick.innerHTML = "";
  const current = !game.last_trick || (game.trick.length > 0 && !game.over);
  const plays = current ? game.trick : game.last_trick.plays;
  trick.classList.toggle("past", !current);
  $("trick-title").textContent = current ? `Trick ${game.tricks_played + 1}` : "Last trick";
  $("trick-type").textContent = current ? `Allowed: ${TRICK_TYPES[game.trick_type]}` : "";
  $("trick-type").classList.toggle("hidden", !current);
  plays.forEach((pl, i) => {
    const leading = current && i === game.trick_leader;
    const box = el("div", "play" + (leading ? " leading" : "") + (current && i >= freshFrom ? " fresh" : ""));
    const by = el("div", "by", who(pl.seat));
    if (leading && plays.length > 1) by.append(el("span", "lead-tag", "leading"));
    const lastOfSeat = !plays.slice(i + 1).some((q) => q.seat === pl.seat);
    if (!current && lastOfSeat) by.append(el("span", "gain", `+${game.last_trick.taken[pl.seat]}`));
    box.append(by, cardRow(pl.cards, true));
    trick.append(box);
  });
  if (!plays.length) trick.append(el("span", "hint", `${who(game.current)} to lead`));
}

function renderHand() {
  const mine = game.players[game.seat];
  const myTurn = game.current === game.seat;
  $("me").classList.toggle("turn", myTurn);
  $("me-name").textContent = seatName(game.seat);
  $("me-points").textContent = `${mine.points} pts`;
  $("me-reserve").textContent = mine.reserve_size ? `${mine.reserve_size} cards in reserve` : "";
  $("turn-hint").textContent = game.over ? "" : myTurn ? "Your turn" : `Waiting for ${seatName(game.current)}…`;

  const hand = $("hand");
  hand.innerHTML = "";
  let group = null;
  let lastRank = null;
  game.hand.forEach((rank, i) => {
    if (rank !== lastRank) {
      group = el("div", "group");
      hand.append(group);
      lastRank = rank;
    }
    const c = el("button", `card r${rank}`, String(rank));
    const selected = selection.includes(i);
    c.classList.toggle("sel", selected);
    c.setAttribute("aria-pressed", String(selected));
    c.setAttribute("aria-label", `Card ${rank}`);
    c.onclick = () => {
      selection = selected ? selection.filter((j) => j !== i) : [...selection, i];
      renderHand();
    };
    group.append(c);
  });
  if (!game.hand.length) hand.append(el("span", "hint", "No cards left"));

  const match = matchingLegal();
  const legal = $("legal");
  legal.innerHTML = "";
  // mark the plays that beat the trick, unless they all do (or none do)
  const beating = game.legal.filter((l) => l.beats).length;
  const markBeats = game.trick.length > 0 && beating > 0 && beating < game.legal.length;
  for (const kind of ["single", "set", "run"]) {
    const options = game.legal.filter((l) => l.kind === kind);
    if (!options.length) continue;
    const row = el("div", "legal-group");
    row.append(el("span", "legal-kind", KIND_NAMES[kind]));
    options.forEach((l) => {
      const b = el("button", "chip" + (match && match.action === l.action ? " match" : "") + (markBeats && l.beats ? " beats" : ""));
      b.setAttribute("aria-label", `${l.cards.join(" ")}${markBeats && l.beats ? ", beats the trick" : ""}`);
      b.title = markBeats && l.beats ? "Beats every play so far" : "";
      l.cards.forEach((r) => b.append(card(r, true)));
      b.onclick = () => { selectCombo(l); renderHand(); };
      b.ondblclick = () => play(l.action);
      row.append(b);
    });
    legal.append(row);
  }
  const btn = $("play-selected");
  btn.disabled = !match;
  btn.textContent = match ? `Play ${cardsText(match.cards)}` : myTurn ? "Select cards" : "Play";
  btn.onclick = () => match && play(match.action);
  $("legal-help").classList.toggle("hidden", !myTurn);
}

function renderSide() {
  // scores, best first
  const scores = $("scores");
  scores.innerHTML = "";
  [...game.players.keys()]
    .sort((a, b) => game.players[b].points - game.players[a].points)
    .forEach((seat) => {
      const li = el("li", seat === game.seat ? "mine" : "");
      li.append(el("span", null, who(seat)), el("span", "points", String(game.players[seat].points)));
      scores.append(li);
    });

  // cards I haven't seen, per rank
  const unseen = $("unseen");
  unseen.innerHTML = "";
  $("unseen-box").classList.toggle("hidden", !game.unseen);
  (game.unseen || []).forEach((count, i) => {
    const cell = el("div", "unseen-cell");
    cell.append(card(i + 1, true), el("span", null, `×${count}`));
    cell.setAttribute("aria-label", `${count} unseen ${i + 1}s`);
    unseen.append(cell);
  });

  // trick log, newest first
  const log = $("log");
  log.innerHTML = "";
  game.log.slice().reverse().forEach((t, k) => {
    const li = el("li");
    li.append(el("div", "log-title", `Trick ${game.log.length - k}`));
    t.plays.forEach((pl, i) => {
      const line = el("div", "log-line");
      line.append(el("span", null, who(pl.seat)), cardRow(pl.cards, true));
      const lastOfSeat = !t.plays.slice(i + 1).some((q) => q.seat === pl.seat);
      if (lastOfSeat) line.append(el("span", "gain", `+${t.taken[pl.seat]}`));
      li.append(line);
    });
    log.append(li);
  });
  if (!game.log.length) log.append(el("li", "hint", "Finished tricks appear here."));
}

function renderTable() {
  const n = game.players.length;
  const spectator = game.seat === null;
  const base = spectator ? 0 : game.seat;
  renderOpponents(spectator, base, n);
  renderMiddle();
  $("me").classList.toggle("hidden", spectator);
  if (!spectator) renderHand();
  renderSide();

  // game over
  $("game-over").classList.toggle("hidden", !game.over);
  if (game.over) {
    const winners = game.winners || [];
    $("over-title").textContent = winners.includes(game.seat)
      ? winners.length > 1 ? "You tied for the win!" : "You win!"
      : `${winners.map(seatName).join(" and ")} win${winners.length > 1 ? "" : "s"}`;
    const standings = $("standings");
    standings.innerHTML = "";
    [...game.players.keys()]
      .sort((a, b) => game.players[b].points - game.players[a].points)
      .forEach((seat) => {
        const li = el("li", winners.includes(seat) ? "win" : "");
        li.append(el("span", null, who(seat)), el("span", "points", `${game.players[seat].points} pts`));
        standings.append(li);
      });
    $("next-game").classList.toggle("hidden", !me.host);
    $("over-hint").textContent = me.host ? "" : "Waiting for the host to start the next game.";
    if (me.host && document.activeElement !== $("next-game")) $("next-game").focus();
  }
}

// ---------- views ----------
function render() {
  const inRoom = me.code && room && room.code === me.code;
  const view = !inRoom ? "home" : room.phase === "lobby" || !game ? "lobby" : "table";
  for (const v of ["home", "lobby", "table"]) $(v).classList.toggle("hidden", v !== view);
  $("room-badge").classList.toggle("hidden", !inRoom);
  if (inRoom) {
    $("room-badge").innerHTML = "";
    $("room-badge").append(document.createTextNode("Room "), el("b", null, room.code));
    if (room.spectators) $("room-badge").append(document.createTextNode(` · ${room.spectators} watching`));
  }
  if (view === "lobby") renderLobby();
  if (view === "table") renderTable();
}

// Keyboard: 1-6 add a card of that rank to the selection, Enter plays it, Escape clears it.
document.addEventListener("keydown", (e) => {
  if (!game || game.seat === null || $("table").classList.contains("hidden")) return;
  const tag = document.activeElement.tagName;
  if (tag === "INPUT" || tag === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Enter" && tag !== "BUTTON") {
    const match = matchingLegal();
    if (match) { e.preventDefault(); play(match.action); }
  } else if (e.key === "Escape") {
    selection = [];
    renderHand();
  } else if (/^[1-6]$/.test(e.key)) {
    const rank = +e.key;
    const i = game.hand.findIndex((r, j) => r === rank && !selection.includes(j));
    if (i >= 0) { selection = [...selection, i]; renderHand(); }
  }
});

async function main() {
  $("name").value = localStorage.getItem("rainbow.name") || "";
  try {
    presets = await (await fetch("/api/agents")).json();
  } catch {
    toast("Couldn't load the agent list");
  }
  setupHome();
  $("rules-btn").onclick = () => $("rules").showModal();
  $("lobby-start").onclick = () => send({ t: "start" });
  $("next-game").onclick = () => send({ t: "start" });
  $("lobby-leave").onclick = leave;
  $("over-leave").onclick = leave;
  $("copy-link").onclick = () => {
    navigator.clipboard.writeText($("share-link").value).then(() => toast("Link copied"), () => $("share-link").select());
  };
  const code = new URLSearchParams(location.search).get("room");
  if (code) {
    $("join-code").value = code.toUpperCase();
    // join straight away only if we know the player's name; otherwise let them enter it
    if (localStorage.getItem("rainbow.name")) {
      me.code = code.toUpperCase();
      me.token = localStorage.getItem(`rainbow.token.${me.code}`);
    } else {
      $("name").focus();
    }
  }
  connect();
  render();
}

main();
