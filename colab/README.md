# Neural Chess Lab

An educational, self-learning chess neural network designed for Google Colab. Neural Chess Lab combines a compact PyTorch policy/value network, Monte Carlo Tree Search (MCTS), self-play training, weight inspection, and an interactive browser interface.

> This project demonstrates the AlphaZero-style learning loop at classroom scale. It is designed for experimentation and understanding—not immediate grandmaster-level play.

## Download

Use the latest notebook:

- `Neural_Chess_Lab_Colab_v2.ipynb`

## Quick start in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Neural_Chess_Lab_Colab_v2.ipynb`.
3. Select **Runtime → Change runtime type**.
4. Choose a **T4 GPU** or another available GPU.
5. Select **Runtime → Run all**.
6. At the bottom of the notebook, click **Open browser lab ↗**.

The optional notebook self-play and training cells are disabled by default so that **Run all** starts quickly. Enable their checkboxes when you want to run experiments directly in the notebook, or use the browser lab.

## What students can do

- Play chess against the current neural network.
- Analyze arbitrary positions using FEN notation.
- Compare raw network predictions with MCTS-improved policies.
- Generate new training examples through self-play.
- Train the policy and value heads on replay memory.
- Follow policy loss, value loss, total loss, and replay growth.
- Inspect every named parameter tensor and its statistics.
- Change an individual weight and observe the effect.
- Save, download, and restore model checkpoints.
- Access the model through documented HTTP/JSON endpoints.

## System overview

The learning cycle is:

1. The neural network predicts a move policy and position value.
2. MCTS uses those predictions to explore possible continuations.
3. The root visit counts become an improved move policy.
4. The model plays games against itself using that policy.
5. Positions, improved policies, and final outcomes enter replay memory.
6. Gradient descent trains the network on sampled replay positions.
7. The improved network guides the next round of search and self-play.

## Neural network

The default model is a compact residual policy/value network with approximately 913,000 trainable parameters.

### Input

Each position becomes an `18 × 8 × 8` tensor:

| Planes | Information |
|---:|---|
| 0–5 | White pawn, knight, bishop, rook, queen, and king |
| 6–11 | Black pawn, knight, bishop, rook, queen, and king |
| 12 | Side to move |
| 13–16 | Castling rights |
| 17 | Normalized half-move clock |

### Outputs

- **Policy head:** logits over 4,672 AlphaZero-style move actions.
- **Value head:** a number from `-1` to `+1`, evaluated from the side-to-move perspective.

Illegal moves are masked before probabilities are calculated.

### Loss

The training objective combines policy cross-entropy and value mean-squared error:

```text
total loss = policy loss + value loss
```

The policy target comes from MCTS visit counts. The value target is the final self-play result: win (`+1`), draw (`0`), or loss (`-1`).

## Monte Carlo Tree Search

MCTS repeatedly performs four operations:

1. **Selection:** choose a child using its current value and PUCT exploration bonus.
2. **Expansion:** create children for the legal moves at a new leaf.
3. **Evaluation:** ask the network for the leaf value, or use the exact terminal result.
4. **Backup:** propagate the value toward the root, reversing its sign after every ply.

Self-play adds Dirichlet noise at the root to encourage opening diversity. Analysis and human play do not add this noise.

## Browser lab

The notebook starts a FastAPI service and displays it through Colab's authenticated port proxy. HTML, CSS, and JavaScript are embedded in the notebook, so no separate web project is required.

### Play & Analyze

- Select a piece and destination to make a move.
- Let the model make a move for either side.
- Paste a FEN position into the position microscope.
- Use `0` simulations to inspect the raw neural-network policy.
- Use a positive simulation count to inspect the search-improved policy.

### Self-Train

Configure:

- number of self-play games;
- MCTS simulations per move;
- gradient-update steps.

The operation runs in a background thread. The page reports job progress, replay-buffer size, games completed, gradient steps, and recent losses.

### Weights

The parameter report includes:

- tensor name and shape;
- number of scalar parameters;
- mean and standard deviation;
- minimum and maximum;
- L2 norm.

To edit a scalar, supply the exact tensor name, a comma-separated index, and a finite value. Download a checkpoint before weight-editing experiments.

## HTTP API

The browser uses the same JSON API available to student projects.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/status` | Device, model, replay, training, and job status |
| `POST` | `/api/analyze` | Analyze a FEN with the raw network or MCTS |
| `POST` | `/api/play` | Submit a human move and receive a model reply |
| `POST` | `/api/model-move` | Ask the model to move for the side to move |
| `POST` | `/api/self-train` | Start a background self-play and training cycle |
| `GET` | `/api/weights` | Return live parameter statistics |
| `POST` | `/api/weights/edit` | Change one scalar parameter |
| `GET` | `/api/checkpoint` | Download a PyTorch checkpoint |
| `POST` | `/api/reset` | Reinitialize the model and clear learning history |
| `GET` | `/docs` | Open FastAPI's interactive API documentation |

Example analysis request:

```javascript
const response = await fetch("api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    simulations: 24,
    top_k: 8
  })
});

const analysis = await response.json();
console.log(analysis);
```

When running through the Colab proxy, browser code should use relative URLs such as `api/analyze`.

## Configuration

The `LabConfig` dataclass near the beginning of the notebook exposes the main experiment settings:

| Setting | Default | Effect |
|---|---:|---|
| `channels` | 64 | Width of the residual network |
| `residual_blocks` | 4 | Depth of the shared feature tower |
| `learning_rate` | 0.002 | Size of optimizer updates |
| `batch_size` | 32 | Replay examples per update |
| `replay_capacity` | 15,000 | Maximum stored positions |
| `simulations` | 24 | Default MCTS simulations per move |
| `c_puct` | 1.5 | Exploration strength during search |
| `temperature_moves` | 20 | Number of high-diversity opening plies |
| `max_game_plies` | 160 | Self-play cutoff |
| `seed` | 7 | Random seed for repeatability |

Changing `channels` or `residual_blocks` requires rebuilding the model. Existing checkpoints must use the same architecture.

## Checkpoints

A checkpoint contains:

- model parameters;
- optimizer state;
- configuration values;
- training history;
- self-play game history.

Use the browser's **Download checkpoint** button for a local copy. The final notebook cell also contains optional Google Drive mounting and checkpoint examples.

Colab runtime storage is temporary. Save important checkpoints to Google Drive or download them before the runtime disconnects.

## Dependency compatibility

The latest notebook uses dependency ranges compatible with current Colab packages such as Gradio, python-fasthtml, and Google ADK:

```bash
pip install --upgrade \
  "python-chess>=1.999,<2" \
  "fastapi>=0.133,<1" \
  "starlette>=1.3.1,<2" \
  "uvicorn>=0.34,<1" \
  "nest-asyncio>=1.6,<2"
```

If you previously ran the older notebook with FastAPI `0.115.6` and Starlette `0.41.3`, restart the Colab session and run `Neural_Chess_Lab_Colab_v2.ipynb` from the top.

## Troubleshooting

### No GPU is detected

Select **Runtime → Change runtime type → GPU**, reconnect, and run the notebook again. The project also runs on CPU, but MCTS self-play will be considerably slower.

### The browser link does not open

- Confirm that the server-start cell completed.
- Keep the Colab runtime connected.
- Rerun the server cell if the runtime restarted.
- Do not reuse an old proxy URL after reconnecting.

### Port 8000 is already in use

Rerun the server cell once. It attempts to stop the previous notebook server before starting a new one. If another application owns the port, restart the runtime.

### Training appears slow

- Confirm that `device=cuda` is printed near the top.
- Reduce MCTS simulations per move.
- Reduce the self-play game count.
- Lower `max_game_plies` during short classroom demonstrations.
- Remember that MCTS performs many individual network evaluations; GPU batching is an advanced extension.

### Loss decreases but play does not improve

Training loss measures fit to the current self-play data, not chess strength. Compare checkpoints in controlled matches, alternate colors, use identical search budgets, and report uncertainty across enough games.

### Weight editing causes unstable outputs

Restore a checkpoint or reset the model. Edit one scalar at a time and avoid extreme values. A parameter participates in many downstream activations, so a local edit can affect many positions.

## Suggested experiments

### Introductory

1. Compare raw policy predictions with 16, 32, and 64 MCTS simulations.
2. Train for 20 gradient steps and explain each loss component.
3. Change one early convolutional weight and repeat a fixed-FEN analysis.

### Intermediate

1. Compare two learning rates using the same seed and replay examples.
2. Disable Dirichlet noise and measure opening diversity.
3. Add horizontal-reflection augmentation, including the correct move transformation.
4. Create an arena that alternates colors between two checkpoints.

### Advanced

1. Batch MCTS leaf evaluation for better GPU utilization.
2. Add a transposition table using Zobrist hashes.
3. Gate new checkpoints with arena matches against a champion model.
4. Add side-to-move canonicalization and compare sample efficiency.
5. Estimate playing strength with confidence intervals instead of relying on loss.

## Limitations

The teaching implementation intentionally omits several production or research-scale features:

- distributed self-play workers;
- batched asynchronous search;
- transposition caching;
- symmetry augmentation;
- opening books and tablebases;
- checkpoint arena gating;
- calibrated resignation;
- ELO evaluation infrastructure.

These omissions keep the full system understandable inside one notebook and provide useful extension projects for advanced students.

## Responsible interpretation

- Sharing AlphaZero's high-level loop does not imply AlphaZero-level results.
- Self-play can reinforce existing blind spots.
- Short training runs have high variance.
- Record the random seed, GPU, number of games, simulations, updates, and evaluation method.
- Use held-out positions and controlled opponent matches to evaluate progress.

## License and classroom use

This notebook is intended as an educational starting point. Before redistributing or publishing a derivative, add the license and attribution terms required by your institution or project.
