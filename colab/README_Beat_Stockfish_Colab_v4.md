
# ♟️ Beat Stockfish — Google Colab Auto-play + Decision Logger

A Google Colab chess environment where you can either play Stockfish manually or let a **Colab Bot auto-play** your side. The notebook is designed for experimentation, self-play data collection, and later neural-network training.

## Notebook

`Beat_Stockfish_Chess_Colab_Autoplay_Logged.ipynb`

## Main features

- Manual play against Stockfish using SAN or UCI.
- Auto-play mode: Colab Bot vs Stockfish Opponent.
- Adjustable auto-play delay/speed.
- Separate Stockfish skill and think-time controls for the Colab Bot and the opponent.
- Win-friendly opponent mode with configurable weaker MultiPV move selection.
- Start/pause auto-play.
- Per-run maximum ply count to prevent unbounded execution.
- Hints, undo, and new-game controls.
- Immediate per-ply persistence.
- Optional Google Drive storage.
- One-click ZIP export of the full session.

## Recommended auto-play setup for the Colab Bot to win more often

| Setting | Suggested value |
|---|---:|
| Mode | Auto-play |
| Colab Bot skill | 20 |
| Colab Bot think time | 0.20–0.50 s |
| Opponent skill | 0–3 |
| Win-friendly opponent | On |
| Opponent blunder chance | 50–75% |
| Opponent think time | 0.05–0.15 s |
| Auto delay | 0.05–0.25 s |

The **auto delay** is only the pause between plies. Actual speed is also limited by the two engine think-time settings.

## Storage and logging

The notebook creates one directory per session.

If Google Drive is mounted before the game cell:

```text
/content/drive/MyDrive/ChessColabLogs/<session_id>/
```

Otherwise:

```text
/content/chess_logs/<session_id>/
```

Local `/content` storage disappears when the runtime is deleted, so either mount Drive or use **Download Logs ZIP**.

### `decisions.jsonl`

The detailed source of truth. Each decision contains:

- session ID and game ID
- decision number and UTC timestamp
- actor (`Human`, `Colab Bot`, or `Stockfish Opponent`)
- board FEN before and after the move
- move SAN and UCI
- all MultiPV candidate moves returned for engine decisions
- candidate rank
- centipawn/mate evaluation
- partial principal variation
- chosen candidate rank
- selection reason (best move, win-friendly weaker choice, manual input, etc.)
- current skill / think-time / blunder / auto-play settings
- game-over state and result

The JSONL file is flushed and `fsync`'d after every decision.

### `moves.csv`

Compact per-ply rows for analysis in pandas, spreadsheets, or ML preprocessing.

### `events.jsonl`

Non-move events such as:

- new/reset games
- setting changes
- auto-play start/pause/errors
- hints
- illegal manual move attempts
- undo operations
- session shutdown

### `pgn/<game_id>.pgn`

A recoverable PGN snapshot for each game. The PGN file is rewritten after **every move**, so an interrupted session retains the game up to the most recently saved ply.

### `results.csv`

Completed-game summaries, including result, termination reason, number of plies, and mode.

## Auto-play flow

```text
                 ┌──────────────────┐
                 │ Current position │
                 └────────┬─────────┘
                          │
               side to move?
                 ┌────────┴────────┐
                 │                 │
                 v                 v
          Colab Bot          Stockfish Opponent
          best MultiPV       best or handicapped
                 │                 │
                 └────────┬────────┘
                          v
                    chosen move
                          │
                          v
                  push board state
                          │
                          v
           save JSONL + CSV + PGN
                          │
                          v
                    delay slider
                          │
                          └──────> next ply
```

## Why save candidate moves?

A final PGN tells you **what was played**, but not **why it was played**. `decisions.jsonl` preserves the alternatives evaluated by Stockfish and makes the data useful later for:

- policy-network training
- imitation learning
- value-model datasets
- ranking-loss experiments
- blunder classification
- self-play analysis
- reconstructing the decision process

For neural training, a useful record becomes roughly:

```text
(position, candidate policies/evaluations, selected move, final outcome)
```

## Speed

Total time per ply is roughly:

```text
engine think time + auto delay + notebook/UI overhead
```

For fast data generation, reduce the delay toward `0` and use short engine think times. For higher-quality training data, increase think time and accept slower game generation.

## Important note

This notebook uses one Stockfish process for both engine-controlled sides, changing the requested skill before each analysis. That is compact and Colab-friendly. If you later want maximum throughput, the next step is to use multiple engine worker processes and run several games in parallel.

## Future neural-network path

The logs are deliberately structured so they can feed the chess neural-network project:

```text
Saved self-play games
        ↓
Position / move dataset
        ↓
Policy + value network
        ↓
Neural player
        ↓
MCTS / Stockfish teacher
        ↓
New self-play games
        ↓
Replay buffer
        ↓
Retraining
```


## Play Hint — v3

v3 adds a dedicated **▶ Play Hint** button in Manual mode.

### Hint vs Play Hint

**Hint** analyzes the current position and displays the recommended move.

**▶ Play Hint** does the full assisted workflow in one click:

1. Analyze the current position with the Colab Hint engine.
2. Collect up to 8 Stockfish MultiPV candidates.
3. Select candidate rank #1.
4. Immediately play that move for your side.
5. Persist the full analysis and decision.
6. Let the Stockfish opponent make its normal reply.

### Logging

A Play Hint move is intentionally distinguishable from a manual human move.

It is stored with:

- actor: `Human via Play Hint`
- selection reason: `play_hint_best_candidate`
- event: `play_hint_requested`
- FEN before the move
- chosen SAN and UCI
- candidate rank
- centipawn or mate score
- MultiPV candidate list
- principal variations
- updated PGN and move logs

Because the normal durable decision logger is used, the move is flushed to storage immediately after it is played.


## v4 — Visible Auto-play Board Updates

v4 fixes the auto-play display so the chess board visibly advances one move at a time in Google Colab.

### What changed

The previous auto-play implementation was already asynchronous, but fast engine calls and notebook output batching could make multiple board updates appear at once.

v4 explicitly yields control back to the Jupyter/Colab browser event loop after **every rendered ply**.

The sequence is now:

```text
Current board
    ↓
Show "thinking"
    ↓
Calculate exactly one move
    ↓
Save/log the decision
    ↓
Render updated board
    ↓
Yield to Colab/browser
    ↓
Keep position visible for Move delay
    ↓
Calculate next move
```

### Move delay

The old `Auto delay` control is now **Move delay**.

- Minimum: `0.10 s`
- Default: `0.35 s`
- Maximum: `3.00 s`

Recommended values:

| Goal | Move delay |
|---|---:|
| Very fast but still visible | 0.10–0.20 s |
| Easy to watch | 0.30–0.60 s |
| Slow study mode | 1.0–2.0 s |

### Live Auto-play Status

v4 adds a small live status line showing whether auto-play is:

- starting,
- thinking,
- displaying a ply,
- paused,
- stopped,
- at the run limit,
- or finished.

### Logging

The visual update changes do **not** reduce logging.

Each move is still durably saved before auto-play proceeds to the next ply, including:

- FEN before/after
- SAN/UCI move
- actor
- candidate list
- evaluation
- principal variation
- chosen rank
- decision reason
- PGN snapshot
- CSV/JSONL decision logs
- events
- final result

The move is logged first, rendered second, and only then does the next auto-play iteration begin.
