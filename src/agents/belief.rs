//! Hand prediction: a network trained by `python/train_belief.py` maps an observation to the
//! expected card counts of every hidden hand and reserve, which bias determinization towards
//! deals that fit what has happened so far.

use std::sync::Arc;

use crate::agents::dmc::Mlp;
use crate::constants::{CARDS_PER_RANK, NUM_RANKS};
use crate::game::encoding::{HIDDEN_SIZE, HIST_OBS_SIZE, OBS_SIZE};
use crate::game::{DealWeights, Observation};

#[derive(Debug, Clone)]
pub struct Belief {
    net: Arc<Mlp>,
}

impl Belief {
    /// The network reads an observation, or one with the public-history features appended
    /// (`train_belief.py --history`), told apart by its input width.
    ///
    /// # Errors
    /// If the network doesn't map an observation to the hidden-information layout.
    pub fn new(net: Arc<Mlp>) -> Result<Self, String> {
        if ![OBS_SIZE, HIST_OBS_SIZE].contains(&net.inputs()) || net.outputs() != HIDDEN_SIZE {
            return Err(format!(
                "belief network must map {OBS_SIZE} (or {HIST_OBS_SIZE}) inputs to {HIDDEN_SIZE} outputs, got {} -> {}",
                net.inputs(),
                net.outputs()
            ));
        }
        Ok(Self { net })
    }

    /// Predicted card counts for every hidden container, as determinization weights. The
    /// network predicts count / 10 per rank for each relative seat's hand then reserve; the
    /// set-aside cards get whatever unknown cards the predictions leave over.
    pub fn weights(&self, obs: &Observation) -> DealWeights {
        let out = if self.net.inputs() == HIST_OBS_SIZE {
            let mut x = [0.0; HIST_OBS_SIZE];
            obs.encode_with_history(&mut x);
            self.net.forward(&x)
        } else {
            self.net.forward(&obs.encoded())
        };
        let n = obs.num_players();
        let scale = f32::from(CARDS_PER_RANK);
        let mut w = DealWeights::default();
        let mut predicted = [0.0f32; NUM_RANKS];
        for seat in 0..n {
            let rel = (seat + n - obs.observer()) % n;
            let block = &out[rel * 2 * NUM_RANKS..][..2 * NUM_RANKS];
            for r in 0..NUM_RANKS {
                w.hands[seat][r] = (block[r] * scale).max(0.0);
                w.reserves[seat][r] = (block[NUM_RANKS + r] * scale).max(0.0);
                if seat != obs.observer() {
                    predicted[r] += w.hands[seat][r];
                }
                predicted[r] += w.reserves[seat][r];
            }
        }
        let unknown = obs.unknown().counts();
        for r in 0..NUM_RANKS {
            w.aside[r] = (f32::from(unknown[r]) - predicted[r]).max(0.0);
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn weights_drive_consistent_deals() {
        let net = Arc::new(crate::agents::dmc::test_mlp(
            &[(OBS_SIZE, 8), (8, HIDDEN_SIZE)],
            3,
        ));
        let belief = Belief::new(net).unwrap();
        let mut rng = SmallRng::seed_from_u64(1);
        for n in [2, 3, 5] {
            let g = GameState::new(n, 0, &mut rng);
            let obs = g.observe(1);
            let w = belief.weights(&obs);
            assert_eq!(obs.determinize_weighted(&w, &mut rng).observe(1), obs);
        }
        let bad = Arc::new(crate::agents::dmc::test_mlp(&[(OBS_SIZE, 3)], 0));
        assert!(Belief::new(bad).is_err());
    }
}
