
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
