//! Deck sampling for training episodes.

use crownforge_core::{CardId, Ruleset, rules::CardKind};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Troop,
    Building,
    Spell,
    Other,
}

pub fn classify(kind: &CardKind) -> Class {
    match kind {
        CardKind::Troop { .. } => Class::Troop,
        CardKind::Building { .. } => Class::Building,
        CardKind::Spell(_)
        | CardKind::StrikingSpell(_)
        | CardKind::CentreStrikeSpell(_)
        | CardKind::RollingSpell(_)
        | CardKind::AreaSpell { .. } => Class::Spell,
        CardKind::Variant(_) | CardKind::Mirror(_) => Class::Other,
    }
}

/// SplitMix64. Wrapping arithmetic is intentional.
#[derive(Clone, Copy)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

pub fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone)]
pub struct DeckSampler {
    /// Cards random decks may draw from, with cost, class and ability-unit count.
    cards: Vec<(CardId, u8, Class, u8)>,
    pool: Vec<[CardId; 8]>,
    /// Chance that a deck comes from the pool instead of a random draw.
    pub pool_prob: f64,
    pub mirror: bool,
}

impl DeckSampler {
    pub fn new(rules: &Ruleset, allowed: &[CardId], pool: Vec<[CardId; 8]>, mirror: bool) -> Self {
        let cards = allowed
            .iter()
            .map(|&card| {
                let def = rules.card(card);
                (
                    card,
                    def.cost,
                    classify(&def.kind),
                    ability_units(rules, card),
                )
            })
            .collect();
        Self {
            cards,
            pool,
            pool_prob: 1.0,
            mirror,
        }
    }

    pub fn card_count(&self) -> usize {
        self.cards.len()
    }

    pub fn sample(&self, rng: &mut Rng) -> [[CardId; 8]; 2] {
        let first = self.sample_one(rng);
        let second = if self.mirror {
            first
        } else {
            self.sample_one(rng)
        };
        [first, second]
    }

    fn sample_one(&self, rng: &mut Rng) -> [CardId; 8] {
        let from_pool = (rng.next() >> 11) as f64 / (1_u64 << 53) as f64;
        if !self.pool.is_empty() && (self.cards.len() < 8 || from_pool < self.pool_prob) {
            return self.pool[rng.below(self.pool.len())];
        }
        let mut order: Vec<usize> = (0..self.cards.len()).collect();
        let mut best = [CardId::Knight; 8];
        for _ in 0..64 {
            // Partial Fisher-Yates: the first eight entries are a uniform sample.
            for i in 0..8 {
                let j = i + rng.below(order.len() - i);
                order.swap(i, j);
            }
            let picked = &order[..8];
            for (slot, &index) in best.iter_mut().zip(picked) {
                *slot = self.cards[index].0;
            }
            let count = |class: Class| picked.iter().filter(|&&i| self.cards[i].2 == class).count();
            let cost: u32 = picked.iter().map(|&i| u32::from(self.cards[i].1)).sum();
            // Keep decks playable: some spells, mostly troops, sane average cost.
            // One champion per deck, as in the game. The simulator also has only
            // two ability slots per side and panics on a third ability unit.
            let champions: u32 = picked.iter().map(|&i| u32::from(self.cards[i].3)).sum();
            let playable = champions <= 1
                && (1..=3).contains(&count(Class::Spell))
                && count(Class::Building) <= 2
                && count(Class::Troop) >= 4
                && (21..=37).contains(&cost);
            if playable {
                return best;
            }
        }
        // No playable draw: fall back to a deck the simulator always accepts.
        CardId::DEFAULT_DECK
    }
}

/// Units with an ability button that one card deploys.
pub fn ability_units(rules: &Ruleset, card: CardId) -> u8 {
    match &rules.card(card).kind {
        CardKind::Troop { members, .. } => members
            .iter()
            .filter(|member| rules.unit(member.unit).ability.is_some())
            .count() as u8,
        _ => 0,
    }
}
