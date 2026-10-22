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
function matchingLegal() {
  if (!game || !game.legal.length || !selection.length) return null;
  const ranks = selection.map((i) => game.hand[i]).sort((a, b) => a - b).join();
  return game.legal.find((l) => l.cards.join() === ranks) || null;
}

function play(action) {
  selection = [];
  send({ t: "play", action });
}

function renderTable() {
  const n = game.players.length;
  const base = game.seat === null ? 0 : game.seat;
  const spectator = game.seat === null;

  // opponents, clockwise from the seat after mine
  const opps = $("opponents");
  opps.innerHTML = "";
  for (let k = spectator ? 0 : 1; k < n; k++) {
    const seat = (base + k) % n;
    const p = game.players[seat];
    const s = room.seats[seat] || {};
    const box = el("div", "opp" + (game.current === seat ? " turn" : ""));
    const who = el("div", "who");
    const name = el("span", game.current === seat && s.kind === "agent" ? "thinking" : "", seatName(seat));
    who.append(name, el("span", "kind", s.kind === "agent" ? "agent" : s.connected === false ? "away" : ""));
    const backs = el("div", "backs");
    for (let i = 0; i < p.hand_size; i++) backs.append(el("div", "back"));
    const meta = el("div", "meta");
    meta.append(
      el("span", null, `${p.hand_size} card${p.hand_size === 1 ? "" : "s"}${p.reserve_size ? ` + ${p.reserve_size} reserve` : ""}`),
      el("span", "points", `${p.points} pts`),
    );
    box.append(who, backs, meta);
    opps.append(box);
  }

  // piles
  const piles = $("piles");
  piles.innerHTML = "";
  game.piles.forEach((v, i) => {
    const pile = el("div", "pile");
    pile.append(el("b", null, String(v)), el("small", null, ORDINALS[i]));
    piles.append(pile);
  });
  if (!game.piles.length) piles.append(el("span", "hint", "none"));

  // trick: the current one, or the last finished one while a new trick hasn't started
  const trick = $("trick");
  trick.innerHTML = "";
  const current = !game.last_trick || (game.trick.length > 0 && !game.over);
  const plays = current ? game.trick : game.last_trick.plays;
  trick.classList.toggle("past", !current);
  $("trick-title").textContent = current ? `Trick ${game.tricks_played + 1}` : "Last trick";
  $("trick-type").textContent = current ? TRICK_TYPES[game.trick_type] : "";
  plays.forEach((pl, i) => {
    const box = el("div", "play");
    const by = el("div", "by", pl.seat === game.seat ? "You" : seatName(pl.seat));
    const lastOfSeat = !plays.slice(i + 1).some((q) => q.seat === pl.seat);
    if (!current && lastOfSeat) {
      const gain = game.last_trick.taken[pl.seat];
      by.append(document.createTextNode(" "), el("span", "gain", `+${gain}`));
    }
    box.append(by, cardRow(pl.cards, true));
    trick.append(box);
  });
  if (!plays.length) trick.append(el("span", "hint", "Waiting for the lead…"));

  // my hand
  $("me").classList.toggle("hidden", spectator);
  if (!spectator) {
    const mine = game.players[game.seat];
    const myTurn = game.current === game.seat;
    $("me").classList.toggle("turn", myTurn);
    $("me-name").textContent = seatName(game.seat);
    $("me-points").textContent = `${mine.points} pts`;
    $("me-reserve").textContent = mine.reserve_size ? `${mine.reserve_size} cards in reserve` : "";
    $("turn-hint").textContent = game.over ? "" : myTurn ? "Your turn" : `Waiting for ${seatName(game.current)}`;

    const hand = $("hand");
    hand.innerHTML = "";
    hand.classList.toggle("waiting", !myTurn);
    let group = null;
    let lastRank = null;
    game.hand.forEach((rank, i) => {
      if (rank !== lastRank) {
        group = el("div", "group");
        hand.append(group);
        lastRank = rank;
      }
      const c = card(rank);
      if (selection.includes(i)) c.classList.add("sel");
      c.onclick = () => {
        selection = selection.includes(i) ? selection.filter((j) => j !== i) : [...selection, i];
        renderTable();
      };
      group.append(c);
    });
    if (!game.hand.length) hand.append(el("span", "hint", "No cards left"));

    const match = matchingLegal();
    const legal = $("legal");
    legal.innerHTML = "";
    game.legal.forEach((l) => {
      const b = el("button", match && match.action === l.action ? "match" : "");
      b.title = `Play ${l.label}`;
      l.cards.forEach((r) => b.append(card(r, true)));
      b.onclick = () => play(l.action);
      legal.append(b);
    });
    if (myTurn && game.legal.length) {
      legal.prepend(el("span", "hint", "Play:"));
    }
    $("play-selected").disabled = !match;
    $("play-selected").onclick = () => match && play(match.action);
  }

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
        const li = el("li", winners.includes(seat) ? "win" : "",
          `${seat === game.seat ? "You" : seatName(seat)}: ${game.players[seat].points} points`);
        standings.append(li);
      });
    $("next-game").classList.toggle("hidden", !me.host);
    $("over-hint").textContent = me.host ? "" : "Waiting for the host to start the next game.";
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

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && game && document.activeElement.tagName !== "INPUT") {
    const match = matchingLegal();
    if (match) play(match.action);
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
