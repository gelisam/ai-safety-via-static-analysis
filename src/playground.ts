/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import { Activations } from "./activation";
import { BIT_RANGES } from "./range";
import {
  State,
} from "./state";
import {Example2D, shuffle, xyToBits, classifyParityData} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

// Helper function for formatting numbers
function formatNumber(num: number): string {
    // Round to 1 decimal place, but show .0
    let fixed = num.toFixed(1);
    // Avoid -0.0
    if (fixed === "-0.0") {
        return "0.0";
    }
    return fixed;
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "bit7": {f: (x, y) => xyToBits(x, y)[0] ? 1 : 0, label: "bit7"},
  "bit6": {f: (x, y) => xyToBits(x, y)[1] ? 1 : 0, label: "bit6"},
  "bit5": {f: (x, y) => xyToBits(x, y)[2] ? 1 : 0, label: "bit5"},
  "bit4": {f: (x, y) => xyToBits(x, y)[3] ? 1 : 0, label: "bit4"},
  "bit3": {f: (x, y) => xyToBits(x, y)[4] ? 1 : 0, label: "bit3"},
  "bit2": {f: (x, y) => xyToBits(x, y)[5] ? 1 : 0, label: "bit2"},
  "bit1": {f: (x, y) => xyToBits(x, y)[6] ? 1 : 0, label: "bit1"},
  "bit0": {f: (x, y) => xyToBits(x, y)[7] ? 1 : 0, label: "bit0"},
};

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();
state.fastUpdates = true; // Hardcode fast updates

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let minLoss = Number.MAX_VALUE;
let epochsSinceMinLoss = 0;
const stoppingConditionEpochs = 50;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777"]); // Only one color for training loss

const LEARNING_RATES = [10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001];

function updateLearningRateDisplay(newRate: number, highlight = true) {
    const learningRateValues = document.getElementById('learning-rate-values');
    if (!learningRateValues) return;

    const learningRates = Array.from(learningRateValues.children).map(d => parseFloat(d.innerHTML));
    const newIndex = learningRates.indexOf(newRate);

    if (newIndex !== -1) {
        const currentTransform = learningRateValues.style.transform;
        const newTransform = `translateY(-${newIndex * 20}px)`;

        if (currentTransform === newTransform) {
            return;
        }

        learningRateValues.style.transform = newTransform;

        if (highlight) {
            learningRateValues.classList.add('highlight');
            setTimeout(() => {
                learningRateValues.classList.remove('highlight');
            }, 1200);
        }
    }
}

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    state.seed = Math.floor(Math.random() * 900000 + 100000).toString();
    state.serialize();
    userHasInteracted();
    generateData();
    reset();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  let safetySlider = d3.select("#safetySlider").on("input", function() {
    state.safetyImportance = +this.value / 100;
    d3.select("#safetyValue").text(this.value + "%");
    state.serialize();
    userHasInteracted();
  });
  safetySlider.property("value", state.safetyImportance * 100);
  d3.select("#safetyValue").text(formatNumber(state.safetyImportance * 100) + "%");

  // Seed controls
  d3.select("#applyUserSeed").on("click", () => {
    const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
    const newSeed = userSeedInput.value;
    if (newSeed && newSeed.trim() !== "") {
      state.seed = newSeed.trim();
      state.serialize(); // Save the new seed
      userHasInteracted();
      Math.seedrandom(state.seed); // Seed the RNG with the user's input
      updateSeedDisplay(); // Show the user's seed
      generateDataPointsOnly(); // Generate new data points using this seed
      reset(); // Reset the network (it will use the now-seeded RNG)
    }
  });
  // Add Enter key listener for the user seed input
  const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
  userSeedInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default form submission if any
      document.getElementById("applyUserSeed").click(); // Trigger click on apply button
    }
  });
  // Initial display of the seed
  updateSeedDisplay();
}

function updateSeedDisplay() {
  const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
  if (userSeedInput) {
    userSeedInput.value = state.seed;
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let outputNode = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(outputNode.output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss, iteration number and loss chart.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain]);

  // Conditionally skip the rest of the UI updates for performance.
  if (state.fastUpdates && !firstStep) {
    return;
  }
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    result.push(inputName);
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    input.push(INPUTS[inputName].f(x, y));
  }
  return input;
}

function updateLearningRate(loss: number) {
  if (loss <= 0.10) {
    state.learningRate = 0.01;
  } else if (loss <= 0.25) {
    state.learningRate = 0.03;
  } else if (loss <= 0.40) {
    state.learningRate = 0.1;
  } else {
    state.learningRate = 0.3;
  }
  updateLearningRateDisplay(state.learningRate);
}

function getMisclassifiedCount(): number {
  let misclassified = 0;
  // All data is used for training, so we can just use trainData.
  for (const point of trainData) {
    const input = constructInput(point.x, point.y);
    const output = nn.forwardProp(network, input).output;
    const prediction = output >= 0 ? 1 : -1;
    if (prediction !== point.label) {
      misclassified++;
    }
  }
  return misclassified;
}

function getIsSafeInTheory(): boolean {
  const outputNode = nn.getOutputNode(network);
  return outputNode.outputRange[1] < 0.0;
}

function getOutputRange(): [number, number] {
    const outputNode = nn.getOutputNode(network);
    if (outputNode && outputNode.outputRange) {
        return outputNode.outputRange;
    }
    return [0, 0]; // Should not happen if network is trained
}

function reportTrainingComplete() {
  const misclassifiedCount = getMisclassifiedCount();
  const outputRange = getOutputRange();
  const safeInTheory = getIsSafeInTheory();

  const reportMessage = [
    "Training stopped: fixed point reached.",
    "",
    `i. The number of misclassified inputs: ${misclassifiedCount}`,
    `ii. The calculated range of the output: [${formatNumber(outputRange[0])}, ${formatNumber(outputRange[1])}]`,
    `iv. Whether the model is safe in theory: ${safeInTheory}`
  ].join("\n");

  alert(reportMessage);
}

function oneStep(): void {
  iter++;
  shuffle(trainData);
  // The training is done in batches of 10.
  let batchSize = 10;
  for (let i = 0; i < trainData.length / batchSize; i++) {
    let batch = trainData.slice(i * batchSize, (i + 1) * batchSize);
    batch.forEach(point => {
      let input = constructInput(point.x, point.y);
      nn.forwardProp(network, input);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
    });

    // Theory propagation
    nn.forwardPropRanges(network, BIT_RANGES);
    nn.backPropRanges(network, [-1, -1], nn.Errors.SQUARE);

    // Update weights
    nn.updateWeights(network, state.learningRate, state.safetyImportance);
  }

  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateLearningRate(lossTrain);

  // Check for early stopping condition.
  const roundedLoss = parseFloat(lossTrain.toFixed(3));
  if (roundedLoss < minLoss) {
    minLoss = roundedLoss;
    epochsSinceMinLoss = 0;
  } else {
    epochsSinceMinLoss++;
  }

  if (epochsSinceMinLoss >= stoppingConditionEpochs) {
    player.pause();
    reportTrainingComplete();
    // Reset to avoid repeated alerts if the user continues stepping.
    epochsSinceMinLoss = 0;
  }

  // Update the ranges one last time to match the updated weights.
  nn.forwardPropRanges(network, BIT_RANGES);
  updateUI();
}

function reset(onStartup=false, hardcodeWeightsOption?:boolean) { // hardcodeWeightsOption is now optional
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  // Reset early stopping variables
  minLoss = Number.MAX_VALUE;
  epochsSinceMinLoss = 0;

  // Set learning rate to 0.3 and update UI
  state.learningRate = 0.3;
  updateLearningRateDisplay(state.learningRate, false);

  // Determine if weights should be hardcoded
  // Priority:
  // 1. Explicit hardcodeWeightsOption if provided (e.g. during initial parity setup)
  // 2. If state.seed is "0"
  // 3. Default to false if neither of the above
  const shouldUseHardcodedWeights = hardcodeWeightsOption !== undefined ? hardcodeWeightsOption :
                                   (state.seed === "0");

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  // Default to TANH activation for output layer, as problem type is removed.
  let outputActivation = Activations.TANH;
  network = nn.buildNetwork(shape, Activations.RELU, outputActivation, constructInputIds());

  if (shouldUseHardcodedWeights) {
    // Initialize weights for the parity network
    // network[1][i] is 1 if the bitstring has at least i+1 1s
    for (let i=0; i<network[1].length; i++) {
      for (let j=0; j<network[0].length; j++) {
        network[1][i].inputLinks[j].weight = 1;
      }
      network[1][i].bias = -i;
    }

    // except for the last node, which is 1 if all 4 upper bits are 1
    let i = network[1].length - 1;
    for (let j=0; j<network[0].length; j++) {
      if (j < 4) {
        network[1][i].inputLinks[j].weight = 1;
      } else {
        network[1][i].inputLinks[j].weight = 0;
      }
    }
    network[1][i].bias = -3;

    if (network[2]) {
      // network[2][i] is 1 if the bitstring has exactly i+1 1s
      for (let i=0; i<network[2].length-1; i++) {
        for (let j=0; j<network[1].length; j++) {
          network[2][i].inputLinks[j].weight = 0;
        }
        network[2][i].inputLinks[i].weight = 1;
        if (i+1 < network[1].length) {
          network[2][i].inputLinks[i+1].weight = -2;
        }
        network[2][i].bias = 0;
      }

      // except for the last node, repeats the last node of the previous layer
      let i = network[2].length - 1;
      for (let j=0; j<network[1].length; j++) {
        if (j == network[2].length - 1) {
          network[2][i].inputLinks[j].weight = 1;
        } else {
          network[2][i].inputLinks[j].weight = 0;
        }
      }
      network[2][i].bias = 0;
    }

    if (network[3]) {
      // network[3][0] is 2+ if the bitstring has an odd number of 1s
      // and -2 otherwise
      let i = 0;
      for (let j=0; j<network[2].length - 1; j++) {
        if (j % 2 == 0) {
          network[3][i].inputLinks[j].weight = 4;
        } else {
          network[3][i].inputLinks[j].weight = 0;
        }
      }
      network[3][i].bias = -2;

      // except if the last node is set, in which case we want to produce -2 or
      // less even if the above contributes 4*4 + 4*4 + 4*4 + 4*4 - 2 = 62
      let j = network[2].length - 1;
      network[3][i].inputLinks[j].weight = -64;
    }
  }

  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateLearningRate(lossTrain);
  // Update node ranges after network initialization or weight hardcoding
  nn.forwardPropRanges(network, BIT_RANGES);
  updateUI(true);
  updateSeedDisplay(); // Ensure seed display is current
}

/**
 * Sets up the RNG, updates the seed display, and regenerates data points
 * using the current `state.seed`.
 * This function does NOT modify `state.seed` itself.
 */
function generateData() {
  // state.seed must be set by the caller if a change is intended.
  Math.seedrandom(state.seed);
  updateSeedDisplay(); // Update the displayed seed based on current state.seed
  generateDataPointsOnly(); // Generate points using the now-seeded RNG
}

/**
 * Generates data points (train and test) based on the current state settings
 * (dataset, noise, numSamples) and populates trainData and testData.
 * Assumes Math.random has already been seeded.
 */
function generateDataPointsOnly() {
  let numSamples = Math.pow(2, 8);
  // Problem type is removed, default to state.dataset for data generation
  let generator = classifyParityData;
  let data = generator(numSamples, 0);
  trainData = data;
  testData = data;
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
}

function simulationStarted() {
  parametersChanged = false;
}

makeGUI();
// state.seed is initialized to "0" by state.ts on first load if not in hash.
// generateData() will use this seed.
generateData();
reset(true, true); // true for onStartup, true for hardcodeWeights (because seed is "0")