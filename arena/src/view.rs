//! What one player may see, in that player's own seat frame: exactly the
//! contents of the tournament server's `obs` message, with card names already
//! turned into library ids. The training bridge builds it from the engine and
//! the tournament client from the server's JSON; everything the policy reads is
//! computed from this and nothing else, so the two cannot drift apart.

use crate::arena::HEIGHT;
use crate::engine::{ActionResult, Engine};

pub const ACT_NONE: u8 = 0;
pub const ACT_OK: u8 = 1;
/// Any reason a move was ignored, the server's own ("missed", "malformed") included.
pub const ACT_IGNORED: u8 = 2;

#[derive(Clone, Copy)]
pub struct Ent {
    pub id: u32,
    pub mine: bool,
    pub def: usize,
    pub x: f64,
    pub y: f64,
    pub hp: i32,
    pub max_hp: i32,
    pub deploying: bool,
    pub tower: bool,
}

#[derive(Default)]
pub struct View {
    pub tick: i64,
    pub elixir: f64,
    pub hand: [usize; 4],
    pub next: usize,
    pub last_action: u8,
    /// Own entities first, then the opponent's, oldest first within each.
    pub ents: Vec<Ent>,
}

impl View {
    pub fn sort(&mut self) {
        self.ents.sort_by_key(|e| (!e.mine, e.id));
    }
}

/// The view `Battle.view(seat).observe()` describes.
pub fn from_engine(eng: &Engine, seat: usize, view: &mut View) {
    view.tick = eng.tick;
    view.elixir = eng.elixir[seat].to_f64();
    view.hand = eng.hand(seat);
    view.next = eng.next_card(seat);
    view.last_action = match eng.last_action[seat] {
        ActionResult::None => ACT_NONE,
        ActionResult::Ok => ACT_OK,
        _ => ACT_IGNORED,
    };
    view.ents.clear();
    let height = HEIGHT.to_f64();
    for e in &eng.entities {
        let mut id = e.id;
        let mut y = e.pos.y.to_f64();
        if seat == 1 {
            y = height - y;
            // The server swaps the two tower triples so "my king" has one id from either seat.
            if e.is_tower && (1..=6).contains(&id) {
                id = if id <= 3 { id + 3 } else { id - 3 };
            }
        }
        view.ents.push(Ent {
            id,
            mine: e.owner as usize == seat,
            def: e.def,
            x: e.pos.x.to_f64(),
            y,
            hp: e.hp,
            max_hp: e.max_hp,
            deploying: e.deploy > 0,
            tower: e.is_tower,
        });
    }
    view.sort();
}
