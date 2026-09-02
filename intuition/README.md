# Neural Network Intuition Lab

A pair of standalone, browser-based teaching labs for helping students build intuition for neural networks by changing inputs, architectures, weights, biases, and connections while watching the forward pass update in real time.

## Files

- `interactive_pytorch_network_builder.html` — **Version 1**, a general neural-network intuition lab with several real-world classification examples.
- `neural_network_chess_lab_v2.html` — **Version 2**, a chess-only neural-network lab with an interactive chessboard, candidate moves, chess-specific features, editable weights, a live math trace, and generated PyTorch code.

Both pages are self-contained HTML files. They do not require a web server, package manager, framework, or internet connection.

## How to run

Open either `.html` file in a modern browser such as Chrome, Edge, Firefox, or Safari.

You can also serve the folder locally if you prefer:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Version 1 — General Neural Network Lab

Version 1 introduces the mechanics of a small feed-forward neural network using simple real-world examples such as:

- Predicting whether a house may sell for a high price
- Predicting whether a student may pass a course
- Predicting whether a plant may need watering
- Predicting whether an email may be spam
- Predicting whether bike-rental demand may be high

Students can:

- Change real-world inputs
- Add or remove hidden layers
- Change the number of neurons
- Edit every weight and bias
- Enable or disable individual connections
- See node activations update live
- Follow the arithmetic of a forward pass
- See generated PyTorch code that reflects the current network

## Version 2 — Chess Move Neural Network Lab

Version 2 uses chess as the learning environment.

Students can:

- Manipulate an elegant interactive chessboard
- Select a piece and inspect pseudo-legal moves
- Choose candidate moves and see them scored
- Apply moves, undo moves, and reset positions
- Switch the side to move
- Use board-edit mode to place or remove pieces
- Load several teaching positions
- Compare the network's top candidate moves
- Inspect chess-specific numeric features
- Edit all weights and biases
- Disable individual neural-network connections
- Watch edge thickness and color change with the weights
- Follow a live forward-pass trace
- Generate PyTorch containing the exact current architecture and parameters

### Chess inputs used in Version 2

Version 2 now feeds the **complete current board state** into the neural network using a transparent 74-value vector:

1. **64 board-square inputs** — one number for every square from `a8` through `h1`
2. **Side to move** — `+1` for White and `-1` for Black
3. **Candidate move coordinates** — normalized from-file, from-rank, to-file, and to-rank
4. **5 derived chess signals** — capture value, centrality, material balance, future mobility, and king safety

Square values use a simple signed piece code: empty = `0`; white pawn/knight/bishop/rook/queen/king use positive `1/6 ... 1`; black pieces use the matching negative values. This keeps every piece and every square visible to the network while remaining easy to inspect.

The page includes a live 8×8 input-encoding inspector showing the exact number supplied for every board square.

### Architecture flexibility

Students can add hidden layers without a fixed layer-count cap in the interface. Each hidden layer can be resized up to 32 neurons, and the network visualization scrolls horizontally as the architecture grows.

### Saving and sharing experiments

The chess lab can preserve the complete experiment:

- **Export JSON** downloads the board, side to move, candidate move, architecture, activation, weights, biases, enabled/disabled connections, and move history.
- **Import JSON** restores an exported experiment.
- **Save in browser** stores the current lab using local browser storage.
- **Load saved** restores that browser-local snapshot.

The generated PyTorch uses the current 74-value input vector and the exact weights currently shown in the table.

## Important teaching note

These labs are educational visualizations, not trained production models.

The default weights are hand-chosen to create understandable behavior. In a real machine-learning project, the weights would normally be learned from data using a loss function, backpropagation, and an optimizer.

The central idea is:

```text
real-world state
    ↓
numbers / features
    ↓
weighted sums: z = Σ(weight × input) + bias
    ↓
activation functions
    ↓
prediction / score
```

The generated PyTorch code shows how the same architecture can be represented with `nn.Linear` layers and a `forward()` method.

## Chess limitations

Version 2 is deliberately not a complete chess engine.

Its move generator is **pseudo-legal** and currently does not fully enforce:

- Check and checkmate
- Pins
- Castling
- En passant
- Underpromotion
- Repetition rules
- Fifty-move rule

Pawn promotion is simplified to automatic queen promotion when a move is applied.

This simplification is intentional: the page is designed to teach neural-network intuition rather than chess-engine implementation.

## Suggested classroom activities

### Activity 1 — Change one weight

Choose a candidate chess move and change a single input-to-hidden weight from positive to negative.

Observe:

- The edge changes color
- Its visual thickness changes
- The hidden neuron activation changes
- The move score changes
- The math trace changes
- The generated PyTorch changes

### Activity 2 — Remove a connection

Disable one weight using its checkbox.

Ask students what information the receiving neuron can no longer use.

### Activity 3 — Compare two moves

Select two different candidate chess moves.

Ask:

- Which feature changed the most?
- Which hidden neurons responded most strongly?
- Why did the final score move up or down?

### Activity 4 — Random network

Randomize the weights.

This is a useful demonstration that an untrained neural network does not automatically "understand" the problem.

### Activity 5 — Architecture experiment

Add a hidden layer or increase the number of neurons.

Discuss why more parameters increase representational capacity but do not guarantee a better model.

## PyTorch connection

A dense neural-network layer is represented in PyTorch with:

```python
nn.Linear(in_features, out_features)
```

If a layer has 5 inputs and 4 neurons:

```python
nn.Linear(5, 4)
```

PyTorch stores a weight matrix shaped approximately like:

```text
4 output neurons × 5 input features
```

Each neuron also receives a bias.

During a forward pass:

```python
x = torch.relu(self.fc1(x))
```

During training, PyTorch normally adjusts those weights and biases with backpropagation.

In these labs, the student can adjust them directly by hand.

## Project goal

The project is designed to make neural networks feel less mysterious.

Students should come away understanding that a neural network is built from understandable pieces:

- inputs
- weights
- biases
- weighted sums
- activation functions
- layers
- connections
- outputs

The visualizations connect those pieces to the equivalent PyTorch code.
