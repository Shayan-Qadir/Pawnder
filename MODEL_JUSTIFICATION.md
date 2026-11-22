# Model justification — what it does, why ML is needed, and what we verified

> **Short answer:** Pawnder is not trying to predict the exact next chess position.
> It uses a learned board representation to choose, from the player's own past
> blunders, a drill that resembles the kind of mistake they are currently making.
> On a chronological holdout this beats random, most-recent, and hand-rule
> baselines by a wide margin. It is **offline evidence of useful drill selection**,
> not proof of rating improvement (that would require a longitudinal/A-B study,
> which is out of scope here).

Reproduce with:

```bash
.\.venv\Scripts\python.exe -m src.training.evaluate_usefulness
```

Results are written to `metrics_usefulness.json`.

---

## 1. What the model actually does

**Input**

1. The player's last **5 blunders** from their private profile. Each blunder is:
   - the board FEN encoded into a 128-dimensional vector by a CNN, and
   - 3 context numbers: `clock/600`, `(player_rating - opponent_rating)/400`, `ply/100`.
2. The **embeddings of every FEN in that player's private bank** (the model does not generate positions).

**Output**

- `retrieval_logits`: one score for every stored FEN in the private bank. The score blends
  - a GRU query built from the 5-step history, and
  - cosine similarity between the **most recent blunder's learned board embedding** and each candidate.
- `clock_logits` / `rating_logits`: auxiliary predictions for time-pressure and opponent-strength context.

At serving time the API masks recently served FENs, turns the scores into a distribution, samples one candidate, and returns **that stored candidate's** FEN and metadata. Stockfish strength, clock and Elo come from the chosen scenario, not from the auxiliary heads.

In one sentence: **it looks at how you blundered recently and ranks your own past blunders to decide which one you should practise next.**

---

## 2. Why ML was needed for this problem

A hand rule can look at clock time and opponent rating. It cannot compare the *shape of a chess position* — piece placement, king safety, tactical pattern — against a pool of 50–95 stored positions.

The CNN board encoder is learned from Lichess data and gives each FEN a 128-dimensional representation. Retrieval over that representation is what lets Pawnder say "this old blunder looks like the mistake you just made", instead of "this blunder happened at a similar clock time".

The offline comparison makes the difference concrete:

| Method | Same-failure-category drill | Board similarity to later real blunder |
|---|---:|---:|
| **Pawnder ranker (private model)** | **97.4%** | **0.982** |
| Most-recent scenario | 78.9% | 0.860 |
| Random from private bank | 71.3% | 0.789 |
| Clock/rating hand rule | 71.1% | 0.646 |
| Always the majority category | 71.1% | — |

The ranker also put the actual later blunder in its top 5 **68.9%** of the time, versus a ~6.9% chance level for a 5-of-50…95 shortlist.

---

## 3. How the evaluation was made fair

- **No A/B test was used.** This is an offline ranking evaluation.
- Two real Chess.com users: `bonc_1` (10 held-out pairs, bank of 51) and `1tshassan` (19 held-out pairs, bank of 95).
- For each user, the games were split **chronologically**: the model was fine-tuned from the Lichess-pretrained checkpoint on the earlier 80% and evaluated on the later 20%. The production checkpoints were *not* used for scoring, because they were trained on all games and would leak test labels.
- Baselines use the exact same private bank and the exact same held-out targets.
- A **label-shuffle negative control** drops exact-next-FEN accuracy to **7.9%**, essentially chance (random baseline 7.6%), confirming the exact-index signal depends on real history/target alignment.

---

## 4. Honest limitations

- Exact next-FEN top-1 is **0%**. Predicting the precise future position is too hard for this dataset; that is why the usefulness metric is "right kind of drill" and board similarity, not exact replay.
- The three weakness labels are deterministic (`Time scramble blunder (<45s)`, `Endgame miscalculation`, `Tactical deterioration`) and the private banks are dominated by one or two of them. Category accuracy is therefore a useful proxy, not a full measure of chess improvement. The label-shuffle category number stays high because of that imbalance; only the exact-index negative control is informative.
- The Lichess-pretrained global model and the privately fine-tuned model score the same on category (97.4%). With 40–75 training pairs, personalization comes mainly from the **private bank** plus the **history-conditioned score**; the private checkpoint is the hook for future feedback-driven training rather than the source of the current gain.
- We did **not** run a live user study, so we do not claim "users got better". The claim is narrower and supported by the numbers above: **the model selects more relevant practice positions from a user's own history than simple alternatives.**

---

## 5. One-paragraph version for a presentation

Pawnder turns a user's Chess.com blunders into a private drill bank. A CNN trained on Lichess encodes every position, and a history network scores those positions against the user's recent mistakes; the API then serves one of them as a legal Stockfish practice position. We validated this on a chronological 80/20 split over two real users: the ranker picked a drill from the same failure category as the player's later real blunders 97% of the time, compared with 71–79% for random, most-recent, and clock/rating rules, and its selected positions were far more visually similar to the later blunder (0.982 vs 0.789 random). A label-shuffle control collapses exact prediction to chance. That is our evidence that the ML component is doing useful work; we deliberately do not claim a rating improvement because no A/B/longitudinal test was run.
