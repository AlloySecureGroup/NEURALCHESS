/* global Chess */

"use strict";

const STORAGE_KEY = "myFiveLayerChessBrainV1";

const PIECE_CODE = {
  p: 1.0,
  n: 2.0,
  b: 3.0,
  r: 4.0,
  q: 5.0,
  k: 6.0
};

const MATERIAL_VALUE = {
  p: 1.0,
  n: 3.0,
  b: 3.25,
  r: 5.0,
  q: 9.0,
  k: 0.0
};

const PIECE_SYMBOL = {
  wp: "♙", wn: "♘", wb: "♗", wr: "♖", wq: "♕", wk: "♔",
  bp: "♟", bn: "♞", bb: "♝", br: "♜", bq: "♛", bk: "♚"
};

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

function rand(min, max) {
  return min + Math.random() * (max - min);
}

class NeuralNet {
  constructor(sizes = [64, 32, 24, 16, 8, 1]) {
    this.sizes = sizes.slice();
    this.weights = [];
    this.biases = [];
    this.trainingUpdates = 0;
    this.lastUpdate = null;

    for (let layer = 0; layer < sizes.length - 1; layer += 1) {
      const nIn = sizes[layer];
      const nOut = sizes[layer + 1];
      const scale = 1 / Math.sqrt(nIn);

      const layerWeights = [];

      for (let j = 0; j < nOut; j += 1) {
        const row = [];

        for (let i = 0; i < nIn; i += 1) {
          row.push(rand(-scale, scale));
        }

        layerWeights.push(row);
      }

      this.weights.push(layerWeights);
      this.biases.push(new Array(nOut).fill(0));
    }
  }

  forward(x) {
    const activations = [x.slice()];

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const W = this.weights[layer];
      const b = this.biases[layer];
      const previous = activations[activations.length - 1];
      const current = [];

      for (let j = 0; j < W.length; j += 1) {
        let z = b[j];

        for (let i = 0; i < previous.length; i += 1) {
          z += W[j][i] * previous[i];
        }

        current.push(Math.tanh(z));
      }

      activations.push(current);
    }

    return activations;
  }

  predict(x) {
    return this.forward(x).at(-1)[0];
  }

  // This function changes weights and biases immediately in memory.
  trainOne(x, target, learningRate = 0.01) {
    const activations = this.forward(x);
    const output = activations.at(-1)[0];

    let delta = [
      2 * (output - target) * (1 - output * output)
    ];
    let totalAbsoluteWeightChange = 0;
    let changedWeightCount = 0;

    for (let layer = this.weights.length - 1; layer >= 0; layer -= 1) {
      const W = this.weights[layer];
      const previous = activations[layer];
      let previousDelta = null;

      if (layer > 0) {
        previousDelta = [];

        for (let i = 0; i < previous.length; i += 1) {
          let error = 0;

          for (let j = 0; j < delta.length; j += 1) {
            error += W[j][i] * delta[j];
          }

          error *= 1 - previous[i] * previous[i];
          previousDelta.push(error);
        }
      }

      for (let j = 0; j < W.length; j += 1) {
        for (let i = 0; i < previous.length; i += 1) {
          const weightChange = learningRate * delta[j] * previous[i];
          W[j][i] -= weightChange;
          totalAbsoluteWeightChange += Math.abs(weightChange);
          changedWeightCount += 1;
        }

        this.biases[layer][j] -= learningRate * delta[j];
      }

      if (layer > 0) {
        delta = previousDelta;
      }
    }

    this.trainingUpdates += 1;
    this.lastUpdate = {
      target,
      learningRate,
      outputBefore: output,
      meanAbsWeightChange: changedWeightCount
        ? totalAbsoluteWeightChange / changedWeightCount
        : 0
    };

    return this.lastUpdate;
  }

  toJSON() {
    return {
      sizes: this.sizes,
      weights: this.weights,
      biases: this.biases,
      trainingUpdates: this.trainingUpdates,
      lastUpdate: this.lastUpdate
    };
  }

  static fromJSON(data) {
    if (!data || !Array.isArray(data.sizes) || !Array.isArray(data.weights) || !Array.isArray(data.biases)) {
      throw new Error("That file is not a valid neural-network brain.");
    }

    const net = new NeuralNet(data.sizes);
    net.weights = data.weights;
    net.biases = data.biases;
    net.trainingUpdates = Number.isFinite(Number(data.trainingUpdates))
      ? Number(data.trainingUpdates)
      : 0;
    net.lastUpdate = data.lastUpdate && typeof data.lastUpdate === "object"
      ? data.lastUpdate
      : null;
    return net;
  }
}

function squareNames() {
  const names = [];

  for (let rank = 1; rank <= 8; rank += 1) {
    for (const file of FILES) {
      names.push(file + rank);
    }
  }

  return names;
}

const ALL_SQUARES = squareNames();

function encodeBoard(chess) {
  const values = [];

  for (const square of ALL_SQUARES) {
    const piece = chess.get(square);

    if (!piece) {
      values.push(0);
      continue;
    }

    const code = PIECE_CODE[piece.type] / 6;
    values.push(piece.color === "w" ? code : -code);
  }

  return values;
}

function terminalScore(chess) {
  if (chess.in_checkmate()) {
    return chess.turn() === "w" ? -1 : 1;
  }

  if (chess.game_over()) {
    return 0;
  }

  return null;
}

function teacherScore(chess) {
  const terminal = terminalScore(chess);

  if (terminal !== null) {
    return terminal;
  }

  let score = 0;

  for (const square of ALL_SQUARES) {
    const piece = chess.get(square);

    if (!piece) {
      continue;
    }

    const value = MATERIAL_VALUE[piece.type];
    score += piece.color === "w" ? value : -value;
  }

  if (chess.in_check()) {
    score += chess.turn() === "w" ? -0.25 : 0.25;
  }

  return Math.tanh(score / 5);
}

function evaluatePosition(net, chess) {
  const terminal = terminalScore(chess);

  if (terminal !== null) {
    return terminal;
  }

  return net.predict(encodeBoard(chess));
}


function moveTags(move) {
  const tags = [];

  if (move.captured) tags.push(`captures ${move.captured.toUpperCase()}`);
  if (move.promotion) tags.push(`promotes to ${move.promotion.toUpperCase()}`);
  if (move.flags && (move.flags.includes("k") || move.flags.includes("q"))) tags.push("castles");
  if (move.san && move.san.includes("+")) tags.push("check");
  if (move.san && move.san.includes("#")) tags.push("checkmate");

  return tags;
}

function rankLegalMoves(net, chess) {
  const moves = chess.moves({ verbose: true });
  const side = chess.turn();
  const currentScore = evaluatePosition(net, chess);
  const ranked = [];

  for (const move of moves) {
    const played = chess.move(move);
    const score = evaluatePosition(net, chess);
    const fenAfter = chess.fen();
    chess.undo();

    const whiteDelta = score - currentScore;
    const sideBenefit = side === "w" ? whiteDelta : -whiteDelta;

    ranked.push({
      move: { ...move, san: played && played.san ? played.san : move.san },
      score,
      currentScore,
      whiteDelta,
      sideBenefit,
      fenAfter,
      tags: moveTags(played || move)
    });
  }

  ranked.sort((a, b) => b.sideBenefit - a.sideBenefit);

  const bestBenefit = ranked.length ? ranked[0].sideBenefit : 0;
  const worstBenefit = ranked.length ? ranked[ranked.length - 1].sideBenefit : 0;
  const spread = Math.max(0.000001, bestBenefit - worstBenefit);

  ranked.forEach((candidate, index) => {
    candidate.rank = index + 1;
    candidate.preference = ranked.length === 1
      ? 1
      : (candidate.sideBenefit - worstBenefit) / spread;
  });

  return { side, currentScore, ranked };
}

function chooseMove(net, chess, randomness = 0) {
  const moves = chess.moves({ verbose: true });

  if (moves.length === 0) {
    return null;
  }

  if (randomness > 0 && Math.random() < randomness) {
    return moves[Math.floor(Math.random() * moves.length)];
  }

  const analysis = rankLegalMoves(net, chess);
  return analysis.ranked.length ? analysis.ranked[0].move : null;
}

function randomTrainingMove(chess) {
  const moves = chess.moves({ verbose: true });
  const captures = moves.filter((move) => move.captured);

  if (captures.length > 0 && Math.random() < 0.6) {
    return captures[Math.floor(Math.random() * captures.length)];
  }

  return moves[Math.floor(Math.random() * moves.length)];
}

let net = loadBrain();
let game = new Chess();
let humanColor = "w";
let selectedSquare = null;
let legalTargets = [];
let training = false;
let liveTrainingPositions = [];
let lastMoveInfo = null;
let networkAnimationFrame = null;
let networkAnimationStarted = 0;
let previewCandidate = null;
let latestMoveAnalysis = null;

const boardEl = document.getElementById("board");
const sideSelect = document.getElementById("sideSelect");
const newGameBtn = document.getElementById("newGameBtn");
const statusText = document.getElementById("statusText");
const evalText = document.getElementById("evalText");
const moveHistory = document.getElementById("moveHistory");
const train250Btn = document.getElementById("train250Btn");
const train1000Btn = document.getElementById("train1000Btn");
const selfPlayBtn = document.getElementById("selfPlayBtn");
const saveBtn = document.getElementById("saveBtn");
const exportBtn = document.getElementById("exportBtn");
const importInput = document.getElementById("importInput");
const resetBrainBtn = document.getElementById("resetBrainBtn");
const trainProgress = document.getElementById("trainProgress");
const trainMessage = document.getElementById("trainMessage");
const brainState = document.getElementById("brainState");
const recommendationTurnBadge = document.getElementById("recommendationTurnBadge");
const recommendationSummary = document.getElementById("recommendationSummary");
const playerRecommendations = document.getElementById("playerRecommendations");

const playTabBtn = document.getElementById("playTabBtn");
const weightsTabBtn = document.getElementById("weightsTabBtn");
const playPanel = document.getElementById("playPanel");
const weightsPanel = document.getElementById("weightsPanel");
const weightLayerSelect = document.getElementById("weightLayerSelect");
const refreshWeightsBtn = document.getElementById("refreshWeightsBtn");
const weightStats = document.getElementById("weightStats");
const weightMatrixShape = document.getElementById("weightMatrixShape");
const weightMatrixHelp = document.getElementById("weightMatrixHelp");
const weightsTable = document.getElementById("weightsTable");
const biasCount = document.getElementById("biasCount");
const biasGrid = document.getElementById("biasGrid");

const networkTabBtn = document.getElementById("networkTabBtn");
const networkPanel = document.getElementById("networkPanel");
const networkCanvas = document.getElementById("networkCanvas");
const networkMoveLabel = document.getElementById("networkMoveLabel");
const networkMoveSquares = document.getElementById("networkMoveSquares");
const networkEvalValue = document.getElementById("networkEvalValue");
const networkUpdateCount = document.getElementById("networkUpdateCount");
const networkDeltaValue = document.getElementById("networkDeltaValue");
const networkLearningRate = document.getElementById("networkLearningRate");
const networkLayerSummary = document.getElementById("networkLayerSummary");
const animateNetworkBtn = document.getElementById("animateNetworkBtn");
const refreshNetworkBtn = document.getElementById("refreshNetworkBtn");
const targetScoreInput = document.getElementById("targetScoreInput");
const targetScoreValue = document.getElementById("targetScoreValue");
const learningRateInput = document.getElementById("learningRateInput");
const learningRateValue = document.getElementById("learningRateValue");
const trainCurrentPositionBtn = document.getElementById("trainCurrentPositionBtn");
const manualTrainResult = document.getElementById("manualTrainResult");
const candidateSideLabel = document.getElementById("candidateSideLabel");
const candidateCountLabel = document.getElementById("candidateCountLabel");
const candidateMoveList = document.getElementById("candidateMoveList");
const candidateDetail = document.getElementById("candidateDetail");
const clearCandidatePreviewBtn = document.getElementById("clearCandidatePreviewBtn");
const playCandidateBtn = document.getElementById("playCandidateBtn");
const topInputInfluences = document.getElementById("topInputInfluences");
const topConnectionInfluences = document.getElementById("topConnectionInfluences");
const pulseExplanation = document.getElementById("pulseExplanation");


function showTab(name) {
  const showPlay = name === "play";
  const showNetwork = name === "network";
  const showWeights = name === "weights";

  playTabBtn.classList.toggle("active", showPlay);
  networkTabBtn.classList.toggle("active", showNetwork);
  weightsTabBtn.classList.toggle("active", showWeights);

  playPanel.classList.toggle("active", showPlay);
  networkPanel.classList.toggle("active", showNetwork);
  weightsPanel.classList.toggle("active", showWeights);

  playTabBtn.setAttribute("aria-selected", String(showPlay));
  networkTabBtn.setAttribute("aria-selected", String(showNetwork));
  weightsTabBtn.setAttribute("aria-selected", String(showWeights));

  if (showWeights) {
    renderWeights();
  }

  if (showNetwork) {
    renderNeuralNetwork(true);
  }
}

function hiddenNeuronLabel(layerIndex, neuronIndex, isTarget) {
  if (isTarget && layerIndex === net.weights.length - 1) {
    return "Output";
  }

  const hiddenNumber = isTarget ? layerIndex + 1 : layerIndex;
  return `H${hiddenNumber}-${neuronIndex + 1}`;
}

function sourceLabelsForLayer(layerIndex) {
  if (layerIndex === 0) {
    // These match encodeBoard(): a1..h1, a2..h2, ... a8..h8.
    return ALL_SQUARES.slice();
  }

  return Array.from(
    { length: net.sizes[layerIndex] },
    (_, index) => hiddenNeuronLabel(layerIndex, index, false)
  );
}

function targetLabelsForLayer(layerIndex) {
  return Array.from(
    { length: net.sizes[layerIndex + 1] },
    (_, index) => hiddenNeuronLabel(layerIndex, index, true)
  );
}

function calculateWeightStats(matrix) {
  const values = matrix.flat();
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let absSum = 0;

  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    absSum += Math.abs(value);
  }

  const mean = values.length ? sum / values.length : 0;
  let variance = 0;

  for (const value of values) {
    const difference = value - mean;
    variance += difference * difference;
  }

  variance = values.length ? variance / values.length : 0;

  return {
    count: values.length,
    min,
    max,
    mean,
    meanAbs: values.length ? absSum / values.length : 0,
    stdDev: Math.sqrt(variance),
    maxAbs: Math.max(Math.abs(min), Math.abs(max), 0.000001)
  };
}

function statBox(label, value) {
  return `<div class="stat-box"><span>${label}</span><strong>${value}</strong></div>`;
}

function populateWeightLayerSelect() {
  weightLayerSelect.innerHTML = "";

  for (let layer = 0; layer < net.weights.length; layer += 1) {
    const option = document.createElement("option");
    option.value = String(layer);
    option.textContent = `Layer ${layer + 1}: ${net.sizes[layer]} → ${net.sizes[layer + 1]}`;
    weightLayerSelect.appendChild(option);
  }
}

function renderWeights() {
  if (!weightLayerSelect.options.length || weightLayerSelect.options.length !== net.weights.length) {
    populateWeightLayerSelect();
  }

  let layer = Number(weightLayerSelect.value || 0);

  if (!Number.isInteger(layer) || layer < 0 || layer >= net.weights.length) {
    layer = 0;
    weightLayerSelect.value = "0";
  }

  const matrix = net.weights[layer];
  const biases = net.biases[layer];
  const sources = sourceLabelsForLayer(layer);
  const targets = targetLabelsForLayer(layer);
  const stats = calculateWeightStats(matrix);

  weightStats.innerHTML = [
    statBox("Weights", stats.count.toLocaleString()),
    statBox("Minimum", stats.min.toFixed(6)),
    statBox("Maximum", stats.max.toFixed(6)),
    statBox("Mean", stats.mean.toFixed(6)),
    statBox("Mean |weight|", stats.meanAbs.toFixed(6)),
    statBox("Std. dev.", stats.stdDev.toFixed(6)),
    statBox("Training updates", net.trainingUpdates.toLocaleString()),
    statBox("Last mean |Δw|", net.lastUpdate ? net.lastUpdate.meanAbsWeightChange.toExponential(3) : "—")
  ].join("");

  weightMatrixShape.textContent = `${matrix.length} neurons × ${sources.length} inputs = ${stats.count.toLocaleString()} weights`;
  biasCount.textContent = `${biases.length} biases`;

  if (layer === 0) {
    weightMatrixHelp.textContent = "Columns are board-square inputs (a1 through h8). Rows are neurons in the first hidden layer.";
  } else if (layer === net.weights.length - 1) {
    weightMatrixHelp.textContent = "Columns are neurons from the final hidden layer. The single row feeds the network's White-vs-Black evaluation output.";
  } else {
    weightMatrixHelp.textContent = "Columns are neurons from the previous hidden layer. Rows are neurons in the next hidden layer.";
  }

  const header = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.textContent = "to ↓ / from →";
  headerRow.appendChild(corner);

  for (const label of sources) {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  }

  header.appendChild(headerRow);

  const body = document.createElement("tbody");

  matrix.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.textContent = targets[rowIndex];
    tr.appendChild(rowHeader);

    row.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value.toFixed(5);
      td.title = value.toPrecision(12);

      const strength = Math.min(0.78, 0.08 + (Math.abs(value) / stats.maxAbs) * 0.70);
      td.style.setProperty("--weight-strength", strength.toFixed(3));

      if (value > 0.0000001) {
        td.classList.add("weight-positive");
      } else if (value < -0.0000001) {
        td.classList.add("weight-negative");
      } else {
        td.classList.add("weight-zero");
      }

      tr.appendChild(td);
    });

    body.appendChild(tr);
  });

  weightsTable.replaceChildren(header, body);

  biasGrid.innerHTML = "";

  biases.forEach((value, index) => {
    const cell = document.createElement("div");
    cell.className = "bias-cell";

    const label = document.createElement("span");
    label.textContent = targets[index];

    const number = document.createElement("span");
    number.textContent = value.toFixed(6);
    number.title = value.toPrecision(12);

    cell.append(label, number);
    biasGrid.appendChild(cell);
  });
}

function saveBrain() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(net.toJSON()));
  brainState.textContent = "Brain saved locally";

  if (weightsPanel.classList.contains("active")) {
    renderWeights();
  }

  if (networkPanel.classList.contains("active")) {
    renderNeuralNetwork(false);
  }
}

function loadBrain() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);

    if (!raw) {
      return new NeuralNet();
    }

    return NeuralNet.fromJSON(JSON.parse(raw));
  } catch (error) {
    console.warn("Could not load saved brain:", error);
    return new NeuralNet();
  }
}

function setTrainingUI(active) {
  training = active;
  train250Btn.disabled = active;
  train1000Btn.disabled = active;
  selfPlayBtn.disabled = active;
  newGameBtn.disabled = active;
  sideSelect.disabled = active;
}

function boardCoordinates() {
  const coords = [];

  if (humanColor === "w") {
    for (let rank = 8; rank >= 1; rank -= 1) {
      for (const file of FILES) {
        coords.push(file + rank);
      }
    }
  } else {
    for (let rank = 1; rank <= 8; rank += 1) {
      for (const file of [...FILES].reverse()) {
        coords.push(file + rank);
      }
    }
  }

  return coords;
}



function scoreClass(deltaForSide) {
  if (deltaForSide > 0.025) return "score-good";
  if (deltaForSide < -0.025) return "score-bad";
  return "score-flat";
}

function signed(value, digits = 3) {
  const safe = Number.isFinite(value) ? value : 0;
  return `${safe >= 0 ? "+" : ""}${safe.toFixed(digits)}`;
}

function sideName(side) {
  return side === "w" ? "White" : "Black";
}

function renderMoveTags(tags) {
  return tags.map((tag) => `<span class="move-tag">${tag}</span>`).join("");
}

function playHumanCandidate(candidate) {
  if (!candidate || training || game.game_over() || game.turn() !== humanColor) {
    return;
  }

  liveTrainingPositions.push(encodeBoard(game));
  const beforeEval = evaluatePosition(net, game);
  const move = game.move({
    from: candidate.move.from,
    to: candidate.move.to,
    promotion: candidate.move.promotion || "q"
  });

  if (!move) {
    liveTrainingPositions.pop();
    return;
  }

  previewCandidate = null;
  recordMoveForNetwork(move, humanColor === "w" ? "White" : "Black", beforeEval);
  clearSelection();
  renderBoard();

  if (game.game_over()) {
    learnFromFinishedGame();
    return;
  }

  window.setTimeout(makeAiMove, 120);
}

function renderPlayerRecommendations(analysis = null) {
  recommendationTurnBadge.textContent = `${sideName(game.turn())} to move`;

  if (game.game_over()) {
    recommendationSummary.textContent = "Game over — no legal recommendations.";
    playerRecommendations.innerHTML = '<div class="recommendations-empty">Start a new game to get recommendations.</div>';
    return;
  }

  if (game.turn() !== humanColor) {
    recommendationSummary.textContent = "The AI is choosing its move. Your recommendations will refresh when it is your turn.";
    playerRecommendations.innerHTML = '<div class="recommendations-empty">Waiting for your turn…</div>';
    return;
  }

  const result = analysis || rankLegalMoves(net, game);
  const top = result.ranked.slice(0, 5);
  const best = top[0];

  recommendationSummary.textContent = best
    ? `Best neural recommendation: ${best.move.san} · predicted output ${signed(best.score)} · move benefit ${signed(best.sideBenefit)}`
    : "No legal moves.";

  playerRecommendations.innerHTML = "";

  top.forEach((candidate) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recommendation-move";
    button.style.setProperty("--preference", `${Math.round(candidate.preference * 100)}%`);
    button.innerHTML = `
      <div class="recommendation-topline">
        <span class="recommendation-rank">#${candidate.rank}</span>
        <span class="recommendation-san">${candidate.move.san}</span>
        <span class="recommendation-score ${scoreClass(candidate.sideBenefit)}">${signed(candidate.score)}</span>
      </div>
      <div class="recommendation-meta">
        <span>benefit ${signed(candidate.sideBenefit)}</span>
        <span>${candidate.move.from} → ${candidate.move.to}</span>
        ${renderMoveTags(candidate.tags)}
      </div>`;
    button.addEventListener("click", () => playHumanCandidate(candidate));
    playerRecommendations.appendChild(button);
  });
}

function getCandidatePreviewPosition(candidate) {
  if (!candidate) {
    return { chess: game, moveInfo: lastMoveInfo, candidate: null };
  }

  const preview = new Chess(game.fen());
  const beforeEval = evaluatePosition(net, preview);
  const played = preview.move({
    from: candidate.move.from,
    to: candidate.move.to,
    promotion: candidate.move.promotion || "q"
  });

  if (!played) {
    return { chess: game, moveInfo: lastMoveInfo, candidate: null };
  }

  return {
    chess: preview,
    candidate,
    moveInfo: {
      actor: `${sideName(candidate.move.color)} candidate`,
      san: played.san,
      from: played.from,
      to: played.to,
      beforeEval,
      afterEval: evaluatePosition(net, preview),
      preview: true
    }
  };
}

function renderCandidateAnalysis(analysis) {
  latestMoveAnalysis = analysis;
  candidateSideLabel.textContent = `${sideName(analysis.side)} candidates`;
  candidateCountLabel.textContent = `${analysis.ranked.length} legal moves`;
  candidateMoveList.innerHTML = "";

  analysis.ranked.slice(0, 10).forEach((candidate) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-move";
    if (previewCandidate && previewCandidate.move.from === candidate.move.from && previewCandidate.move.to === candidate.move.to && (previewCandidate.move.promotion || "") === (candidate.move.promotion || "")) {
      button.classList.add("selected-candidate");
    }
    button.style.setProperty("--preference", `${Math.round(candidate.preference * 100)}%`);
    button.innerHTML = `
      <div class="candidate-topline">
        <span class="candidate-rank">#${candidate.rank}</span>
        <span class="candidate-san">${candidate.move.san}</span>
        <span class="candidate-score ${scoreClass(candidate.sideBenefit)}">${signed(candidate.score)}</span>
      </div>
      <div class="candidate-meta">
        <span>side benefit ${signed(candidate.sideBenefit)}</span>
        <span>White Δ ${signed(candidate.whiteDelta)}</span>
        ${renderMoveTags(candidate.tags)}
      </div>`;
    button.addEventListener("click", () => {
      previewCandidate = candidate;
      renderNeuralNetwork(true);
    });
    candidateMoveList.appendChild(button);
  });

  const selected = previewCandidate;
  playCandidateBtn.disabled = !(selected && game.turn() === humanColor && !training && !game.game_over());

  if (!selected) {
    candidateDetail.innerHTML = `
      <span class="candidate-detail-kicker">CURRENT POSITION</span>
      <strong>${sideName(analysis.side)} to move</strong>
      <p>The current output is <b>${signed(analysis.currentScore)}</b>. Select a candidate to compare its result with this baseline and make the neural pulse follow that move.</p>
      <div class="decision-note">The AI chooses rank #1 for its own turn. Your move coach uses the same ranking when it is your turn.</div>`;
    return;
  }

  const rankPercent = analysis.ranked.length > 1
    ? Math.round(100 * (1 - (selected.rank - 1) / (analysis.ranked.length - 1)))
    : 100;
  const favorable = selected.sideBenefit > 0.025
    ? "improves the network's evaluation for the side to move"
    : selected.sideBenefit < -0.025
      ? "reduces the network's evaluation for the side to move"
      : "is evaluated as roughly neutral relative to the current position";

  candidateDetail.innerHTML = `
    <span class="candidate-detail-kicker">PREVIEWING RANK #${selected.rank}</span>
    <strong>${selected.move.san}</strong>
    <p>This candidate ${favorable}. The pulse below is now running on the board <em>after</em> ${selected.move.san}.</p>
    <div class="candidate-metrics">
      <div class="candidate-metric"><span>Before</span><strong>${signed(selected.currentScore)}</strong><small>current position</small></div>
      <div class="candidate-metric"><span>After</span><strong>${signed(selected.score)}</strong><small>candidate output</small></div>
      <div class="candidate-metric"><span>Side benefit</span><strong class="${scoreClass(selected.sideBenefit)}">${signed(selected.sideBenefit)}</strong><small>rank percentile ${rankPercent}%</small></div>
    </div>
    <div class="decision-note">For White, higher output is preferred. For Black, lower output is preferred. “Side benefit” flips the sign automatically so positive always means better for whoever is moving.</div>`;
}

function recordMoveForNetwork(move, actor, beforeEval) {
  if (!move) {
    return;
  }

  lastMoveInfo = {
    actor,
    san: move.san || `${move.from}-${move.to}`,
    from: move.from,
    to: move.to,
    beforeEval,
    afterEval: evaluatePosition(net, game)
  };
}

function strongestActivation(values) {
  if (!values.length) {
    return { index: 0, value: 0 };
  }

  let bestIndex = 0;
  let bestValue = values[0];

  for (let i = 1; i < values.length; i += 1) {
    if (Math.abs(values[i]) > Math.abs(bestValue)) {
      bestIndex = i;
      bestValue = values[i];
    }
  }

  return { index: bestIndex, value: bestValue };
}

function layerActivationLabel(layerIndex, neuronIndex) {
  if (layerIndex === 0) {
    return ALL_SQUARES[neuronIndex] || `Input ${neuronIndex + 1}`;
  }

  if (layerIndex === net.sizes.length - 1) {
    return "Output";
  }

  return `H${layerIndex}-${neuronIndex + 1}`;
}

function renderNetworkLayerSummary(activations) {
  const names = ["Board input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"];

  networkLayerSummary.innerHTML = activations.map((values, layerIndex) => {
    const strongest = strongestActivation(values);
    const meanAbs = values.reduce((sum, value) => sum + Math.abs(value), 0) / Math.max(values.length, 1);
    const activeCount = values.filter((value) => Math.abs(value) > 0.05).length;
    const neuronName = layerActivationLabel(layerIndex, strongest.index);

    return `
      <div class="layer-summary-card">
        <span>${names[layerIndex]}</span>
        <strong>${neuronName} ${strongest.value >= 0 ? "+" : ""}${strongest.value.toFixed(3)}</strong>
        <small>mean |a| ${meanAbs.toFixed(3)} · ${activeCount}/${values.length} active</small>
      </div>`;
  }).join("");
}

function activationColor(value, alpha = 1) {
  const magnitude = Math.min(1, Math.abs(value));

  if (Math.abs(value) < 0.025) {
    return `rgba(100, 116, 139, ${Math.min(alpha, 0.62)})`;
  }

  if (value > 0) {
    return `rgba(94, 234, 212, ${Math.max(0.10, alpha * (0.35 + magnitude * 0.65))})`;
  }

  return `rgba(244, 114, 182, ${Math.max(0.10, alpha * (0.35 + magnitude * 0.65))})`;
}

function makeLayerPositions(width, height, activations) {
  const positions = [];
  const top = 50;
  const bottom = height - 64;
  const centerY = (top + bottom) / 2;

  const inputGridSize = Math.min(width * 0.14, height * 0.62);
  const inputLeft = Math.max(18, width * 0.025);
  const inputTop = centerY - inputGridSize / 2;
  const cell = inputGridSize / 8;
  const inputPositions = [];

  for (let index = 0; index < 64; index += 1) {
    const file = index % 8;
    const rank = Math.floor(index / 8);
    inputPositions.push({
      x: inputLeft + (file + 0.5) * cell,
      y: inputTop + (7 - rank + 0.5) * cell,
      radius: Math.max(2.2, cell * 0.24),
      squareSize: Math.max(3.2, cell * 0.52)
    });
  }
  positions.push(inputPositions);

  const xFractions = [0.30, 0.47, 0.62, 0.76, 0.91];

  for (let layer = 1; layer < activations.length; layer += 1) {
    const count = activations[layer].length;
    const x = width * xFractions[layer - 1];
    const available = bottom - top;
    const spacing = count > 1 ? available / (count - 1) : 0;
    const radius = count >= 28 ? 4.0 : count >= 16 ? 5.0 : count >= 8 ? 6.5 : 10;
    const layerPositions = [];

    for (let i = 0; i < count; i += 1) {
      layerPositions.push({
        x,
        y: count === 1 ? centerY : top + spacing * i,
        radius
      });
    }

    positions.push(layerPositions);
  }

  return positions;
}

function topConnectionsForTarget(weights, sourceActivations, targetIndex, limit) {
  const row = weights[targetIndex];
  const candidates = row.map((weight, sourceIndex) => ({
    sourceIndex,
    contribution: weight * sourceActivations[sourceIndex],
    weight
  }));

  candidates.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return candidates.slice(0, Math.min(limit, candidates.length));
}

function drawNetworkFrame(activations, animationProgress = 1, visualMoveInfo = lastMoveInfo) {
  const canvas = networkCanvas;
  const rect = canvas.getBoundingClientRect();

  if (rect.width < 10 || rect.height < 10) {
    return;
  }

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(rect.width * dpr);
  const pixelHeight = Math.round(rect.height * dpr);

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const width = rect.width;
  const height = rect.height;
  const positions = makeLayerPositions(width, height, activations);

  const vignette = ctx.createRadialGradient(width * 0.54, height * 0.48, 20, width * 0.54, height * 0.48, width * 0.62);
  vignette.addColorStop(0, "rgba(39, 67, 104, 0.12)");
  vignette.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, width, height);

  // Strongest weighted contributions between layers.
  for (let layer = 0; layer < net.weights.length; layer += 1) {
    const sourceActs = activations[layer];
    const targetCount = activations[layer + 1].length;
    const limit = layer === 0 ? 5 : 4;
    const layerPhase = layer / Math.max(1, net.weights.length - 1);
    const pulse = Math.max(0, 1 - Math.abs(animationProgress - layerPhase) * 4.0);

    for (let targetIndex = 0; targetIndex < targetCount; targetIndex += 1) {
      const strongest = topConnectionsForTarget(net.weights[layer], sourceActs, targetIndex, limit);
      const targetPos = positions[layer + 1][targetIndex];

      for (const connection of strongest) {
        const sourcePos = positions[layer][connection.sourceIndex];
        const contribution = connection.contribution;
        const strength = Math.min(1, Math.abs(contribution) * 1.8 + Math.abs(connection.weight) * 0.25);

        ctx.beginPath();
        ctx.moveTo(sourcePos.x, sourcePos.y);
        const controlX = (sourcePos.x + targetPos.x) / 2;
        ctx.bezierCurveTo(controlX, sourcePos.y, controlX, targetPos.y, targetPos.x, targetPos.y);
        ctx.strokeStyle = activationColor(contribution, 0.10 + strength * 0.44 + pulse * 0.25);
        ctx.lineWidth = 0.45 + strength * 1.35 + pulse * 0.5;
        ctx.stroke();
      }
    }
  }

  const labels = ["64 BOARD INPUTS", "H1 · 32", "H2 · 24", "H3 · 16", "H4 · 8", "OUTPUT · 1"];
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "700 11px system-ui, -apple-system, sans-serif";

  for (let layer = 0; layer < positions.length; layer += 1) {
    const layerPositions = positions[layer];
    const labelX = layer === 0
      ? layerPositions.reduce((sum, p) => sum + p.x, 0) / layerPositions.length
      : layerPositions[0].x;
    ctx.fillStyle = "rgba(181, 198, 220, 0.78)";
    ctx.fillText(labels[layer], labelX, 24);
  }

  // Input layer as an 8x8 activation map.
  for (let index = 0; index < positions[0].length; index += 1) {
    const pos = positions[0][index];
    const value = activations[0][index];
    const layerPhase = 0;
    const pulse = Math.max(0, 1 - Math.abs(animationProgress - layerPhase) * 5);
    const size = pos.squareSize + pulse * 1.8;
    ctx.fillStyle = activationColor(value, 0.30 + Math.abs(value) * 0.70);
    ctx.fillRect(pos.x - size / 2, pos.y - size / 2, size, size);

    if (Math.abs(value) > 0.02) {
      ctx.strokeStyle = activationColor(value, 0.85);
      ctx.lineWidth = 0.8;
      ctx.strokeRect(pos.x - size / 2, pos.y - size / 2, size, size);
    }
  }

  // Show the latest chess move directly on the 64-input board.
  if (visualMoveInfo) {
    const fromIndex = ALL_SQUARES.indexOf(visualMoveInfo.from);
    const toIndex = ALL_SQUARES.indexOf(visualMoveInfo.to);

    if (fromIndex >= 0 && toIndex >= 0) {
      const from = positions[0][fromIndex];
      const to = positions[0][toIndex];
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.max(0.001, Math.hypot(dx, dy));
      const ux = dx / length;
      const uy = dy / length;
      const arrowSize = Math.max(5, Math.min(width, height) * 0.012);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = "rgba(247, 215, 128, 0.92)";
      ctx.lineWidth = 2.4;
      ctx.shadowColor = "rgba(247, 215, 128, 0.72)";
      ctx.shadowBlur = 12;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(to.x, to.y);
      ctx.lineTo(
        to.x - ux * arrowSize - uy * arrowSize * 0.65,
        to.y - uy * arrowSize + ux * arrowSize * 0.65
      );
      ctx.lineTo(
        to.x - ux * arrowSize + uy * arrowSize * 0.65,
        to.y - uy * arrowSize - ux * arrowSize * 0.65
      );
      ctx.closePath();
      ctx.fillStyle = "rgba(247, 215, 128, 0.96)";
      ctx.fill();

      for (const point of [from, to]) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(5, point.squareSize * 0.62), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(247, 215, 128, 0.92)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.restore();
    }
  }

  // Hidden + output neurons.
  for (let layer = 1; layer < positions.length; layer += 1) {
    const layerPhase = layer / Math.max(1, positions.length - 1);
    const pulse = Math.max(0, 1 - Math.abs(animationProgress - layerPhase) * 4.5);

    for (let index = 0; index < positions[layer].length; index += 1) {
      const pos = positions[layer][index];
      const value = activations[layer][index];
      const magnitude = Math.min(1, Math.abs(value));
      const radius = pos.radius + magnitude * 2.1 + pulse * 2.4;

      if (magnitude > 0.08 || pulse > 0.15) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 5 + pulse * 4, 0, Math.PI * 2);
        ctx.fillStyle = activationColor(value, 0.04 + magnitude * 0.08 + pulse * 0.08);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = activationColor(value, 0.42 + magnitude * 0.55);
      ctx.fill();
      ctx.strokeStyle = "rgba(226, 235, 247, 0.20)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  // Output value label.
  const outputPos = positions.at(-1)[0];
  const output = activations.at(-1)[0];
  ctx.font = "800 14px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillStyle = activationColor(output, 0.95);
  ctx.textAlign = "center";
  ctx.fillText(`${output >= 0 ? "+" : ""}${output.toFixed(3)}`, outputPos.x, outputPos.y + 31);
}

function animateNetwork(activations, visualMoveInfo = lastMoveInfo) {
  if (networkAnimationFrame !== null) {
    cancelAnimationFrame(networkAnimationFrame);
    networkAnimationFrame = null;
  }

  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion) {
    drawNetworkFrame(activations, 1, visualMoveInfo);
    return;
  }

  networkAnimationStarted = performance.now();
  const duration = 1450;

  const frame = (now) => {
    const elapsed = now - networkAnimationStarted;
    const progress = Math.min(1, elapsed / duration);
    drawNetworkFrame(activations, progress, visualMoveInfo);

    if (progress < 1) {
      networkAnimationFrame = requestAnimationFrame(frame);
    } else {
      networkAnimationFrame = null;
      drawNetworkFrame(activations, 1, visualMoveInfo);
    }
  };

  networkAnimationFrame = requestAnimationFrame(frame);
}


function topBoardInfluences(input, limit = 8) {
  const matrix = net.weights[0];
  const rows = [];

  for (let sourceIndex = 0; sourceIndex < input.length; sourceIndex += 1) {
    const activation = input[sourceIndex];
    if (Math.abs(activation) < 0.000001) continue;

    let signedSum = 0;
    let absoluteSum = 0;
    for (let targetIndex = 0; targetIndex < matrix.length; targetIndex += 1) {
      const contribution = activation * matrix[targetIndex][sourceIndex];
      signedSum += contribution;
      absoluteSum += Math.abs(contribution);
    }

    rows.push({
      square: ALL_SQUARES[sourceIndex],
      activation,
      signedSum,
      absoluteSum
    });
  }

  rows.sort((a, b) => b.absoluteSum - a.absoluteSum);
  return rows.slice(0, limit);
}

function strongestLiveConnections(activations, limit = 10) {
  const rows = [];

  for (let layer = 0; layer < net.weights.length; layer += 1) {
    const matrix = net.weights[layer];
    const sourceActs = activations[layer];

    for (let targetIndex = 0; targetIndex < matrix.length; targetIndex += 1) {
      for (let sourceIndex = 0; sourceIndex < matrix[targetIndex].length; sourceIndex += 1) {
        const weight = matrix[targetIndex][sourceIndex];
        const sourceActivation = sourceActs[sourceIndex];
        const contribution = weight * sourceActivation;
        rows.push({
          layer,
          sourceIndex,
          targetIndex,
          weight,
          sourceActivation,
          contribution
        });
      }
    }
  }

  rows.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return rows.slice(0, limit);
}

function renderDetailedInfluences(input, activations, selectedCandidate) {
  const inputs = topBoardInfluences(input, 7);
  const maxInput = Math.max(0.000001, ...inputs.map((item) => item.absoluteSum));

  topInputInfluences.innerHTML = inputs.length
    ? inputs.map((item) => {
        const pieceCode = Math.round(Math.abs(item.activation) * 6);
        const pieceType = Object.keys(PIECE_CODE).find((key) => PIECE_CODE[key] === pieceCode) || "p";
        const color = item.activation >= 0 ? "w" : "b";
        const symbol = PIECE_SYMBOL[color + pieceType] || "•";
        const strength = Math.round((item.absoluteSum / maxInput) * 100);
        return `
          <div class="influence-row ${item.signedSum < 0 ? "negative" : ""}">
            <div class="influence-topline">
              <span class="influence-name">${symbol} ${item.square}</span>
              <span class="influence-value ${item.signedSum >= 0 ? "score-good" : "score-bad"}">${signed(item.signedSum, 4)}</span>
            </div>
            <div class="influence-bar"><i style="--strength:${strength}%"></i></div>
          </div>`;
      }).join("")
    : '<div class="recommendations-empty">No occupied input squares.</div>';

  const connections = strongestLiveConnections(activations, 8);
  const maxConnection = Math.max(0.000001, ...connections.map((item) => Math.abs(item.contribution)));

  topConnectionInfluences.innerHTML = connections.map((item) => {
    const source = layerActivationLabel(item.layer, item.sourceIndex);
    const target = layerActivationLabel(item.layer + 1, item.targetIndex);
    const strength = Math.round((Math.abs(item.contribution) / maxConnection) * 100);
    return `
      <div class="influence-row ${item.contribution < 0 ? "negative" : ""}">
        <div class="influence-topline">
          <span class="influence-name">${source} → ${target}</span>
          <span class="influence-value ${item.contribution >= 0 ? "score-good" : "score-bad"}">${signed(item.contribution, 4)}</span>
        </div>
        <div class="influence-bar"><i style="--strength:${strength}%"></i></div>
      </div>`;
  }).join("");

  const output = activations.at(-1)[0];
  const side = selectedCandidate ? selectedCandidate.move.color : game.turn();
  let narrative;

  if (selectedCandidate) {
    const benefit = selectedCandidate.sideBenefit;
    narrative = benefit > 0.025
      ? `The move <strong>${selectedCandidate.move.san}</strong> pushes the evaluation in ${sideName(side)}'s preferred direction.`
      : benefit < -0.025
        ? `The move <strong>${selectedCandidate.move.san}</strong> pushes the evaluation against ${sideName(side)} according to this network.`
        : `The move <strong>${selectedCandidate.move.san}</strong> leaves the network evaluation close to its current value.`;
  } else if (lastMoveInfo) {
    const delta = lastMoveInfo.afterEval - lastMoveInfo.beforeEval;
    narrative = `The most recent real move <strong>${lastMoveInfo.san}</strong> changed the White-centric output by ${signed(delta)}.`;
  } else {
    narrative = `This is the network's pulse for the starting/current board. Select a candidate above to compare a hypothetical move.`;
  }

  pulseExplanation.innerHTML = `
    <div>${narrative}</div>
    <span class="explanation-output">final output = tanh(weighted H4 signals + bias) = ${signed(output)}</span>
    <div><strong>Important:</strong> the animation is slowed down for you to see it. The actual forward pass happens almost instantly. Connection brightness is based on live activation × weight; it is not a chess-engine search tree or a probability.</div>`;
}

function renderNeuralNetwork(animate = false) {
  if (game.game_over()) {
    previewCandidate = null;
  }

  const analysis = game.game_over()
    ? { side: game.turn(), currentScore: evaluatePosition(net, game), ranked: [] }
    : rankLegalMoves(net, game);

  if (previewCandidate) {
    const stillExists = analysis.ranked.find((candidate) =>
      candidate.move.from === previewCandidate.move.from &&
      candidate.move.to === previewCandidate.move.to &&
      (candidate.move.promotion || "") === (previewCandidate.move.promotion || "")
    );
    previewCandidate = stillExists || null;
  }

  renderCandidateAnalysis(analysis);

  const visualState = getCandidatePreviewPosition(previewCandidate);
  const visualChess = visualState.chess;
  const visualMoveInfo = visualState.moveInfo;
  const input = encodeBoard(visualChess);
  const activations = net.forward(input);
  const output = activations.at(-1)[0];

  networkEvalValue.textContent = signed(output);
  networkUpdateCount.textContent = net.trainingUpdates.toLocaleString();
  networkDeltaValue.textContent = net.lastUpdate
    ? net.lastUpdate.meanAbsWeightChange.toExponential(3)
    : "0.000000";
  networkLearningRate.textContent = net.lastUpdate
    ? `target ${net.lastUpdate.target.toFixed(2)} · lr ${net.lastUpdate.learningRate.toFixed(3)}`
    : "No update yet";

  if (previewCandidate && visualMoveInfo) {
    networkMoveLabel.textContent = `Preview: ${visualMoveInfo.san}`;
    const delta = visualMoveInfo.afterEval - visualMoveInfo.beforeEval;
    networkMoveSquares.textContent = `${visualMoveInfo.from} → ${visualMoveInfo.to} · White Δ ${signed(delta)}`;
  } else if (lastMoveInfo) {
    networkMoveLabel.textContent = `${lastMoveInfo.actor}: ${lastMoveInfo.san}`;
    const delta = lastMoveInfo.afterEval - lastMoveInfo.beforeEval;
    networkMoveSquares.textContent = `${lastMoveInfo.from} → ${lastMoveInfo.to} · eval Δ ${signed(delta)}`;
  } else {
    networkMoveLabel.textContent = "Starting position";
    networkMoveSquares.textContent = "No move yet";
  }

  renderNetworkLayerSummary(activations);
  renderDetailedInfluences(input, activations, previewCandidate);

  if (animate) {
    animateNetwork(activations, visualMoveInfo);
  } else {
    drawNetworkFrame(activations, 1, visualMoveInfo);
  }
}

function trainCurrentPosition() {
  if (training) {
    manualTrainResult.textContent = "Wait for the current training run to finish.";
    return;
  }

  const target = Math.max(-1, Math.min(1, Number(targetScoreInput.value)));
  const learningRate = Math.max(0.001, Math.min(0.05, Number(learningRateInput.value)));
  const visualState = getCandidatePreviewPosition(previewCandidate);
  const input = encodeBoard(visualState.chess);
  const before = net.predict(input);
  const update = net.trainOne(input, target, learningRate);
  const after = net.predict(input);

  saveBrain();
  updateStatus();
  renderWeights();
  renderPlayerRecommendations();
  renderNeuralNetwork(true);

  const subject = previewCandidate ? `candidate ${previewCandidate.move.san}` : "current position";
  manualTrainResult.textContent =
    `trained ${subject} · update #${net.trainingUpdates.toLocaleString()} · output ${signed(before, 4)} → ${signed(after, 4)} · mean |Δw| ${update.meanAbsWeightChange.toExponential(3)}`;
}

function renderBoard() {
  boardEl.innerHTML = "";

  const coords = boardCoordinates();

  coords.forEach((square, index) => {
    const fileIndex = FILES.indexOf(square[0]);
    const rank = Number(square[1]);
    const isLight = (fileIndex + rank) % 2 === 1;
    const piece = game.get(square);

    const button = document.createElement("button");
    button.className = "square " + (isLight ? "light" : "dark");
    button.dataset.square = square;
    button.setAttribute("aria-label", square);

    if (square === selectedSquare) {
      button.classList.add("selected");
    }

    const target = legalTargets.find((move) => move.to === square);

    if (target) {
      button.classList.add("target");

      if (target.captured) {
        button.classList.add("capture");
      }
    }

    if (piece) {
      const pieceSpan = document.createElement("span");
      pieceSpan.className =
        "piece " + (piece.color === "w" ? "white-piece" : "black-piece");
      pieceSpan.textContent = PIECE_SYMBOL[piece.color + piece.type];
      button.appendChild(pieceSpan);
    }

    const fileLabelNeeded = index >= 56;
    const rankLabelNeeded = index % 8 === 0;

    if (fileLabelNeeded) {
      const fileLabel = document.createElement("span");
      fileLabel.className = "coord-file";
      fileLabel.textContent = square[0];
      button.appendChild(fileLabel);
    }

    if (rankLabelNeeded) {
      const rankLabel = document.createElement("span");
      rankLabel.className = "coord-rank";
      rankLabel.textContent = square[1];
      button.appendChild(rankLabel);
    }

    button.addEventListener("click", () => handleSquareClick(square));
    boardEl.appendChild(button);
  });

  updateStatus();
  renderPlayerRecommendations();

  if (networkPanel.classList.contains("active")) {
    renderNeuralNetwork(false);
  }
}

function handleSquareClick(square) {
  if (training || game.game_over() || game.turn() !== humanColor) {
    return;
  }

  const piece = game.get(square);

  if (selectedSquare === null) {
    if (piece && piece.color === humanColor) {
      selectSquare(square);
    }
    return;
  }

  if (piece && piece.color === humanColor) {
    selectSquare(square);
    return;
  }

  const matchingMove = game.moves({ square: selectedSquare, verbose: true })
    .find((move) => move.to === square);

  if (!matchingMove) {
    clearSelection();
    renderBoard();
    return;
  }

  playHumanCandidate({ move: matchingMove });
}

function selectSquare(square) {
  selectedSquare = square;
  legalTargets = game.moves({
    square,
    verbose: true
  });
  renderBoard();
}

function clearSelection() {
  selectedSquare = null;
  legalTargets = [];
}

function makeAiMove() {
  if (game.game_over() || game.turn() === humanColor) {
    updateStatus();
    return;
  }

  statusText.textContent = "AI thinking…";

  window.setTimeout(() => {
    liveTrainingPositions.push(encodeBoard(game));
    const beforeEval = evaluatePosition(net, game);
    const aiAnalysis = rankLegalMoves(net, game);
    const move = aiAnalysis.ranked.length ? aiAnalysis.ranked[0].move : null;

    if (move) {
      const playedMove = game.move(move);
      recordMoveForNetwork(
        playedMove,
        humanColor === "w" ? "Black AI" : "White AI",
        beforeEval
      );
    }

    renderBoard();

    if (game.game_over()) {
      learnFromFinishedGame();
    }
  }, 40);
}

function learnFromFinishedGame() {
  let result = 0;

  if (game.in_checkmate()) {
    result = game.turn() === "w" ? -1 : 1;
  }

  for (const position of liveTrainingPositions) {
    net.trainOne(position, result, 0.002);
  }

  saveBrain();
  updateStatus();
}

function updateStatus() {
  let text;

  if (game.in_checkmate()) {
    text = game.turn() === "w"
      ? "Checkmate — Black wins"
      : "Checkmate — White wins";
  } else if (game.in_draw()) {
    text = "Draw";
  } else if (game.in_check()) {
    text = game.turn() === "w"
      ? "White is in check"
      : "Black is in check";
  } else {
    text = game.turn() === humanColor
      ? "Your turn"
      : "AI turn";
  }

  statusText.textContent = text;
  evalText.textContent = evaluatePosition(net, game).toFixed(3);

  const history = game.history();
  moveHistory.textContent = history.length
    ? history.map((move, index) => `${index + 1}. ${move}`).join("  ")
    : "No moves yet.";
}

function newGame() {
  game = new Chess();
  humanColor = sideSelect.value;
  liveTrainingPositions = [];
  lastMoveInfo = null;
  previewCandidate = null;
  latestMoveAnalysis = null;
  clearSelection();
  renderBoard();

  if (humanColor === "b") {
    window.setTimeout(makeAiMove, 150);
  }
}

async function trainBootstrap(samples) {
  if (training) {
    return;
  }

  setTrainingUI(true);
  trainProgress.value = 0;
  trainMessage.textContent = `Training on ${samples} examples…`;

  const trainer = new Chess();
  let plies = 0;
  const chunk = 20;

  for (let sample = 1; sample <= samples; sample += 1) {
    const target = teacherScore(trainer);

    net.trainOne(
      encodeBoard(trainer),
      target,
      0.012
    );

    if (trainer.game_over() || plies >= 80) {
      trainer.reset();
      plies = 0;
    } else {
      trainer.move(randomTrainingMove(trainer));
      plies += 1;
    }

    if (sample % chunk === 0 || sample === samples) {
      trainProgress.value = Math.round((sample / samples) * 100);
      trainMessage.textContent =
        `Example ${sample}/${samples} — network ${net.predict(encodeBoard(trainer)).toFixed(3)}`;

      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  saveBrain();
  setTrainingUI(false);
  trainMessage.textContent = `Finished ${samples} bootstrap examples · ${net.trainingUpdates.toLocaleString()} total weight updates.`;
  previewCandidate = null;
  renderBoard();
  renderNeuralNetwork(false);
}

async function selfPlayOneGame() {
  if (training) {
    return;
  }

  setTrainingUI(true);
  trainProgress.value = 0;
  trainMessage.textContent = "Self-play game running…";

  const selfGame = new Chess();
  const history = [];
  const maxPlies = 100;

  for (let ply = 0; ply < maxPlies; ply += 1) {
    if (selfGame.game_over()) {
      break;
    }

    history.push(encodeBoard(selfGame));
    const move = chooseMove(net, selfGame, 0.25);

    if (!move) {
      break;
    }

    selfGame.move(move);
    trainProgress.value = Math.round(((ply + 1) / maxPlies) * 100);

    if (ply % 4 === 0) {
      trainMessage.textContent = `Self-play move ${ply + 1}`;
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  let result;

  if (selfGame.in_checkmate()) {
    result = selfGame.turn() === "w" ? -1 : 1;
  } else if (selfGame.game_over()) {
    result = 0;
  } else {
    result = teacherScore(selfGame);
  }

  for (let i = 0; i < history.length; i += 1) {
    net.trainOne(history[i], result, 0.0025);

    if (i % 10 === 0) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  saveBrain();
  setTrainingUI(false);
  trainProgress.value = 100;
  trainMessage.textContent =
    `Self-play finished · target ${result.toFixed(3)} · ${net.trainingUpdates.toLocaleString()} total weight updates.`;
  previewCandidate = null;
  renderBoard();
  renderNeuralNetwork(false);
}

function exportBrain() {
  const blob = new Blob(
    [JSON.stringify(net.toJSON(), null, 2)],
    { type: "application/json" }
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "chess_brain.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function importBrainFile(file) {
  const text = await file.text();
  const parsed = JSON.parse(text);
  net = NeuralNet.fromJSON(parsed);
  populateWeightLayerSelect();
  saveBrain();
  brainState.textContent = "Imported brain loaded";
  previewCandidate = null;
  renderBoard();
}

train250Btn.addEventListener("click", () => trainBootstrap(250));
train1000Btn.addEventListener("click", () => trainBootstrap(1000));
selfPlayBtn.addEventListener("click", selfPlayOneGame);

saveBtn.addEventListener("click", () => {
  saveBrain();
  trainMessage.textContent = "Brain saved in this browser.";
});

exportBtn.addEventListener("click", exportBrain);

importInput.addEventListener("change", async () => {
  const file = importInput.files && importInput.files[0];

  if (!file) {
    return;
  }

  try {
    await importBrainFile(file);
    trainMessage.textContent = "Brain imported successfully.";
  } catch (error) {
    alert("Could not import that brain: " + error.message);
  } finally {
    importInput.value = "";
  }
});

resetBrainBtn.addEventListener("click", () => {
  if (!confirm("Reset all learned weights and create a brand-new brain?")) {
    return;
  }

  net = new NeuralNet();
  populateWeightLayerSelect();
  localStorage.removeItem(STORAGE_KEY);
  brainState.textContent = "New brain";
  trainProgress.value = 0;
  trainMessage.textContent = "Brain reset.";
  manualTrainResult.textContent = "Brain reset — training update counter returned to zero.";
  lastMoveInfo = null;
  previewCandidate = null;
  renderBoard();
  renderNeuralNetwork(false);
});

playTabBtn.addEventListener("click", () => showTab("play"));
networkTabBtn.addEventListener("click", () => showTab("network"));
weightsTabBtn.addEventListener("click", () => showTab("weights"));
weightLayerSelect.addEventListener("change", renderWeights);
refreshWeightsBtn.addEventListener("click", renderWeights);

animateNetworkBtn.addEventListener("click", () => renderNeuralNetwork(true));
refreshNetworkBtn.addEventListener("click", () => renderNeuralNetwork(false));
trainCurrentPositionBtn.addEventListener("click", trainCurrentPosition);

targetScoreInput.addEventListener("input", () => {
  targetScoreValue.textContent = Number(targetScoreInput.value).toFixed(2);
});

learningRateInput.addEventListener("input", () => {
  learningRateValue.textContent = Number(learningRateInput.value).toFixed(3);
});

clearCandidatePreviewBtn.addEventListener("click", () => {
  previewCandidate = null;
  renderNeuralNetwork(true);
});

playCandidateBtn.addEventListener("click", () => {
  if (previewCandidate) {
    playHumanCandidate(previewCandidate);
  }
});

if (typeof ResizeObserver !== "undefined") {
  const networkResizeObserver = new ResizeObserver(() => {
    if (networkPanel.classList.contains("active")) {
      renderNeuralNetwork(false);
    }
  });
  networkResizeObserver.observe(networkCanvas);
} else {
  window.addEventListener("resize", () => {
    if (networkPanel.classList.contains("active")) {
      renderNeuralNetwork(false);
    }
  });
}

populateWeightLayerSelect();

newGameBtn.addEventListener("click", newGame);

sideSelect.addEventListener("change", () => {
  humanColor = sideSelect.value;
  newGame();
});

brainState.textContent = localStorage.getItem(STORAGE_KEY)
  ? "Saved brain loaded"
  : "New brain";

renderBoard();
renderNeuralNetwork(false);
