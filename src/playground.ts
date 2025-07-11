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
import { ActivationFunction, Activations } from "./activation";
import { BIT_RANGES } from "./range";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  // regularizations, // Removed
  getKeyFromValue
} from "./state";
import {Example2D, shuffle, xyToBits, classifyParityData, bitlength, DataGenerator} from "./dataset";
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

export function generateNetworkCode(network: nn.Node[][], state: State): string {
    if (!network || network.length === 0) return "";

    let codeLines: string[] = [];
    let nodeIdToVarName: {[key: string]: string} = {};

    // Populate nodeIdToVarName and add input ranges
    const inputLayer = network[0];
    for (let j = 0; j < inputLayer.length; j++) {
        const node = inputLayer[j];
        nodeIdToVarName[node.id] = node.id; // e.g. bit0
        if (node.range) {
            codeLines.push(`[${formatNumber(node.range[0])}, ${formatNumber(node.range[1])}] ∋ ${node.id}`);
        }
    }
    if (inputLayer.length > 0) {
        codeLines.push(""); // Add a blank line after input ranges
    }

    for (let i = 1; i < network.length; i++) { // Start from the first hidden layer
        let layer = network[i];
        for (let j = 0; j < layer.length; j++) {
            const node = layer[j];
            let varName: string;
            if (i === network.length - 1) { // Output layer
                varName = "out";
            } else { // Hidden layer
                const layerIndex = i - 1;
                const nodeIndex = j;
                varName = `x${layerIndex}${nodeIndex}`;
            }
            nodeIdToVarName[node.id] = varName;
        }
    }


    for (let i=1; i<network.length; i++) {
      if (i === network.length - 1) { // Output layer
        codeLines.push(""); // Add an extra blank line before the output layer
      }

      let layer = network[i];
      for (let j=0; j<layer.length; j++) {
        const node = layer[j];
        const targetName = nodeIdToVarName[node.id];
        const allTermsArray: string[] = [];

        // Add range string for the current node
        if (node.range) {
          const rangeString = `[${formatNumber(node.range[0])}, ${formatNumber(node.range[1])}] ∋ ${targetName}`;
          if (i === network.length - 1) { // Output layer
            allTermsArray.push(`<span id='output-node-range-and-var-text'>${rangeString}</span>`);
          } else {
            allTermsArray.push(rangeString);
          }
        } else {
          allTermsArray.push(targetName);
        }
        allTermsArray.push(`=`);

        const termsArray: string[] = [];
        let firstTerm = true;
        for (let k=0; k<node.inputLinks.length; k++) {
          const link = node.inputLinks[k];

          // Omitting terms of the form "0.0 * x"
          if (Math.abs(link.weight) < 0.001) continue;

          const sourceName = nodeIdToVarName[link.source.id];
          const weight = link.weight;

          // Omitting 1.0 coefficients
          if (Math.abs(weight - 1.0) < 0.001) {
            if (weight > 0) {
              if (firstTerm) {
                // x10 = x00
                termsArray.push(sourceName);
                firstTerm = false;
              } else {
                // x10 = ... + x00
                termsArray.push("+");
                termsArray.push(sourceName);
              }
            } else {
              if (firstTerm) {
                // x10 = -x00
                termsArray.push(`-${sourceName}`);
                firstTerm = false;
              } else {
                // x10 = ... - x00
                termsArray.push("-");
                termsArray.push(sourceName);
              }
            }
          } else {
            if (weight > 0) {
              if (firstTerm) {
                // x10 = 0.5 * x00
                termsArray.push(formatNumber(weight));
                termsArray.push("*");
                termsArray.push(sourceName);
                firstTerm = false;
              } else {
                // x10 = ... + 0.5 * x00
                termsArray.push("+");
                termsArray.push(formatNumber(weight));
                termsArray.push("*");
                termsArray.push(sourceName);
              }
            } else {
              if (firstTerm) {
                // x10 = -0.5 * x00
                termsArray.push(formatNumber(weight));
                termsArray.push("*");
                termsArray.push(sourceName);
                firstTerm = false;
              } else {
                // x10 = ... - 0.5 * x00
                termsArray.push("-");
                termsArray.push(formatNumber(Math.abs(weight)));
                termsArray.push("*");
                termsArray.push(sourceName);
              }
            }
          }
        }

        // Bias term
        if (Math.abs(node.bias) >= 0.001) {
          if (node.bias > 0.0) {
            if (firstTerm) {
              // x10 = 0.5
              termsArray.push(formatNumber(node.bias));
              firstTerm = false;
            } else {
              // x10 = ... + 0.5
              termsArray.push("+");
              termsArray.push(formatNumber(node.bias));
            }
          } else {
            if (firstTerm) {
              // x10 = -0.5
              termsArray.push(formatNumber(node.bias));
              firstTerm = false;
            } else {
              // x10 = ... - 0.5
              termsArray.push("-");
              termsArray.push(formatNumber(Math.abs(node.bias)));
            }
          }
        }

        let joinedTerms = termsArray.join(" ");
        // If no terms were added (all weights zero, bias zero), represent as relu(0.0)
        if (firstTerm) {
          joinedTerms = "0.0";
        }

        allTermsArray.push(`${node.activation.name}(${joinedTerms})`);
        codeLines.push(allTermsArray.join(" "));
      }
      if (i < network.length - 1) {
        codeLines.push(""); // Blank line after each layer
      }
    }
    return codeLines.join("\n");
}

let mainWidth;

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const DENSITY = 100;

enum HoverType {
  BIAS, WEIGHT
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

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  // ["Regularization", "regularization"], // Removed
  // ["Regularization rate", "regularizationRate"], // Removed
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

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

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-5.3, 5.3];
let heatMap =
    new HeatMap(600, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: false});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777"]); // Only one color for training loss
let recentTrainLosses: number[] = [];
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

function enableFeaturesForDataset(dataset: DataGenerator) {
  // First, disable all feature inputs
  for (let inputName in INPUTS) {
    state[inputName] = false;
  }

  // Enable appropriate features based on dataset
  if (dataset === datasets.parity) {
    // Enable bit1 through bit8 for parity
    for (let i = 0; i < state.numBits; i++) {
      state[`bit${i}`] = true;
    }
    // Set network shape for parity: two layers of 8 neurons each
    state.networkShape = [8, 8];
  } else {
    // Enable x and y for all other datasets
    state.x = true;
    state.y = true;
    // Set network shape for other datasets: 4 neurons, then 2 neurons
    state.networkShape = [4, 2];
  }
  state.numHiddenLayers = state.networkShape.length;
}

function updateCodeDisplay() {
  const codeDisplayElement = document.getElementById("code-display");
  if (codeDisplayElement) {
    // Ensure 'network' and 'state' are the currently updated instances
    // These are typically available in the global scope of playground.ts or passed around
    const codeString = generateNetworkCode(network, state);
    codeDisplayElement.innerHTML = codeString; // Use innerHTML to parse spans
  } else {
    // console.warn("code-display element not found");
    // Warning: element might not be ready in very early stages or if HTML changes.
    // For this specific project, it should generally be available once UI is built.
  }
}

function makeGUI() {
  // Add collapsible section functionality
  initCollapsibleSections();

  d3.select("#reset-button").on("click", () => {
    // Main reset button now generates a new random seed
    state.seed = Math.floor(Math.random() * 900000 + 100000).toString();
    state.serialize();
    userHasInteracted();
    generateData(); // Uses the new random seed
    reset(); // Reset network (will use the new random seed, no hardcoded weights)
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
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

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset = newDataset;
    enableFeaturesForDataset(newDataset);
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;

    // If parity is selected, we want to ensure it starts with the hardcoded seed and weights
    if (newDataset === datasets.parity) {
      state.seed = "0";
      state.serialize(); // Persist this seed if serialization were active
      generateData(); // Uses the "0" seed
      reset(false, true); // Explicitly hardcode weights for parity
    } else {
      // For other datasets, generate a new random seed
      state.seed = Math.floor(Math.random() * 900000 + 100000).toString();
      state.serialize();
      generateData(); // Uses the new random seed
      reset(); // Reset normally
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

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  state.noise = 0;

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }

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

  // Initial display of the seed
  updateSeedDisplay();
}

function updateSeedDisplay() {
  const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
  if (userSeedInput) {
    userSeedInput.value = state.seed;
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

let isHovercardBeingEdited = false;

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        if (!isHovercardBeingEdited) {
          updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
        }
      }).on("mouseleave", function() {
        if (!isHovercardBeingEdited) {
          updateHoverCard(null);
        }
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset(false, state.dataset === datasets.parity);
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Use a fixed width instead of calculating from DOM positions
  // This prevents issues when sections are collapsed
  let padding = 3;
  let width = 600; // features.column's width in the CSS
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2 + 30;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg, plus a bit for the "This is the output from
  // one neuron" tips.
  svg.attr("height", maxY + 100);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    isHovercardBeingEdited = false;
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    isHovercardBeingEdited = true;
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        nn.updateNodeRanges(network, BIT_RANGES);
        updateUI();
      }
    });
    input.on("keydown", () => {
      let keycode = (d3.event as any).keyCode;
      if (keycode === 13 || keycode === 27) {
        updateHoverCard(null, nodeOrLink, coordinates);
      }
    });
    let inputElement = input.node() as HTMLInputElement;
    inputElement.focus();
    inputElement.select();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      if (!isHovercardBeingEdited) {
        updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
      }
    }).on("mouseleave", function() {
      if (!isHovercardBeingEdited) {
        updateHoverCard(null);
      }
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Create network wrapper with predict function for highlighting incorrect predictions
  let networkWrapper = {
    predict: (point: Example2D) => {
      let input = constructInput(point.x, point.y);
      let output = nn.forwardProp(network, input);
      return output >= 0 ? 1 : -1; // Convert continuous output to classification
    }
  };

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  // Update heatmap points with incorrect prediction highlighting
  heatMap.updatePoints(trainData, networkWrapper);
  heatMap.updateTestPoints(state.showTestData ? testData : [], networkWrapper);

  // Draw the top row highlight
  let allVisiblePoints = trainData.slice();
  if (state.showTestData) {
    allVisiblePoints = allVisiblePoints.concat(testData);
  }
  // The networkWrapper created earlier in updateUI has the .predict method
  heatMap.drawTopRowHighlight(allVisiblePoints, networkWrapper);

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

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  // d3.select("#loss-test").text(humanReadable(lossTest)); // Removed
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  updateCodeDisplay(); // Update code display whenever UI refreshes

  // Handle "unsafe in practice" box
  const practiceSafetyContainer = d3.select("#practice-safety-container");
  practiceSafetyContainer.html(""); // Clear previous content

  let isUnsafeInPractice = false;
  let dataToCheck: Example2D[] = trainData.slice();
  if (state.showTestData) {
    dataToCheck = dataToCheck.concat(testData);
  }

  for (const point of dataToCheck) {
    const input = constructInput(point.x, point.y);
    const prediction = nn.forwardProp(network, input);
    const predictedLabel = prediction >= 0 ? 1 : -1; // Discretize output
    if (predictedLabel !== point.label) {
      // Check for critical misclassification (emergency stop ignored)
      // input[0] to input[3] correspond to bit7 to bit4
      const isEmergencyStopPattern = input.slice(0, 4).every(bit => bit === 1);
      if (isEmergencyStopPattern && predictedLabel === 1) { // Predicts incinerate (1) for emergency stop
        isUnsafeInPractice = true;
        break;
      }
    }
  }

  const practiceSafetyBox = practiceSafetyContainer.append("div")
    .attr("id", "practice-safety-box")
    .attr("class", isUnsafeInPractice ? "unsafe" : "safe");

  practiceSafetyBox.append("span")
    .attr("id", "practice-safety-text")
    .attr("class", isUnsafeInPractice ? "unsafe" : "safe")
    .text(isUnsafeInPractice ? "unsafe in practice" : "safe in practice");


  // Handle "unsafe in theory" highlighting
  let outputNode = nn.getOutputNode(network);
  // Target the new span that includes the variable name 'out'
  let outputRangeAndVarSpan = document.getElementById("output-node-range-and-var-text");
  let codeDisplayElement = document.getElementById("code-display");

  // Remove previous theory-safety-box if it exists
  const existingTheorySafetyBox = document.getElementById("theory-safety-box");
  if (existingTheorySafetyBox) {
    existingTheorySafetyBox.remove();
  }

  if (outputNode && outputNode.range && outputRangeAndVarSpan && codeDisplayElement) {
    const isUnsafeInTheory = outputNode.range[1] >= 0.0;
    const codeDisplayRect = codeDisplayElement.getBoundingClientRect();
    // Get bounding rect of the new span "output-node-range-and-var-text"
    const targetRect = outputRangeAndVarSpan.getBoundingClientRect();

    const theorySafetyBox = document.createElement("div");
    theorySafetyBox.id = "theory-safety-box";
    theorySafetyBox.className = isUnsafeInTheory ? "unsafe" : "safe";

    theorySafetyBox.style.top = (targetRect.top - codeDisplayRect.top + codeDisplayElement.scrollTop) + "px";
    theorySafetyBox.style.left = (targetRect.left - codeDisplayRect.left + codeDisplayElement.scrollLeft - 2) + "px";
    theorySafetyBox.style.width = (targetRect.width + 6) + "px";
    theorySafetyBox.style.height = (targetRect.height + 3) + "px";

    const theorySafetyText = document.createElement("span");
    theorySafetyText.id = "theory-safety-text";
    theorySafetyText.className = isUnsafeInTheory ? "unsafe" : "safe";
    theorySafetyText.textContent = (isUnsafeInTheory ? "unsafe" : "safe") + " in theory";

    theorySafetyBox.appendChild(theorySafetyText);

    const sectionContent = codeDisplayElement.parentElement;
    if (sectionContent) {
        if (window.getComputedStyle(sectionContent).position !== 'relative' && window.getComputedStyle(sectionContent).position !== 'absolute' && window.getComputedStyle(sectionContent).position !== 'fixed') {
            sectionContent.style.position = "relative";
        }
        sectionContent.appendChild(theorySafetyBox);
    }
  }

  lineChart.addDataPoint([lossTrain]); // Only add training loss
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  let currentLearningRate = state.learningRate;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, currentLearningRate, 0, state.activation); // Regularization rate is now 0
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);

  // Zigzag detection logic
  if (trainData.length > 0) { // Only run if there's training data
    recentTrainLosses.push(lossTrain);
    if (recentTrainLosses.length > 10) {
      recentTrainLosses.shift(); // Keep only the last 10 losses
    }

    if (recentTrainLosses.length === 10 && iter >= state.cooldownActiveUntilIter) {
      let increases = 0;
      for (let k = 1; k < recentTrainLosses.length; k++) {
        if (recentTrainLosses[k] > recentTrainLosses[k - 1]) {
          increases++;
        }
      }

      if (increases >= 5) {
        console.log(`Zigzag detected at iteration ${iter}. Increases: ${increases}. Current LR: ${state.learningRate}`);
        const currentLRIndex = LEARNING_RATES.indexOf(state.learningRate);
        if (currentLRIndex > 0) { // Ensure it's not already the smallest
          const newLearningRate = LEARNING_RATES[currentLRIndex - 1];
          state.learningRate = newLearningRate;
          // Update the UI dropdown
          (d3.select("#learningRate").node() as HTMLSelectElement).value = newLearningRate.toString();
          console.log(`Learning rate reduced to ${newLearningRate}`);
          state.cooldownActiveUntilIter = iter + 30; // Start cooldown

          // Highlight the learning rate dropdown input element
          const lrInputElement = d3.select("#learningRate").node() as HTMLElement;
          if (lrInputElement) {
            lrInputElement.classList.add("lr-input-highlight");
            setTimeout(() => {
              lrInputElement.classList.remove("lr-input-highlight");
            }, 1000); // Highlight for 1 second
          }
        } else {
          console.log("Learning rate already at minimum.");
        }
      }
    }
  }

  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false, hardcodeWeightsOption?:boolean) { // hardcodeWeightsOption is now optional
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  // Set learning rate to 0.3 and update UI
  state.learningRate = 0.3;
  d3.select("#learningRate").property("value", "0.3");
  recentTrainLosses = []; // Also clear recent losses on reset
  state.cooldownActiveUntilIter = 0; // Reset cooldown


  // Determine if weights should be hardcoded
  // Priority:
  // 1. Explicit hardcodeWeightsOption if provided (e.g. during initial parity setup)
  // 2. If state.seed is "0"
  // 3. Default to false if neither of the above
  const shouldUseHardcodedWeights = hardcodeWeightsOption !== undefined ? hardcodeWeightsOption :
                                   (state.seed === "0");

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  if (shouldUseHardcodedWeights) {
    for (let inputName in INPUTS) {
      state[inputName] = false;
    }

    for (let i = 0; i < state.numBits; i++) {
      state[`bit${i}`] = true;
    }
  }
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  // Default to TANH activation for output layer, as problem type is removed.
  let outputActivation = Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      null, constructInputIds(), state.initZero); // Regularization is now null

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
  // Update node ranges after network initialization or weight hardcoding
  nn.updateNodeRanges(network, BIT_RANGES);
  drawNetwork(network);
  updateUI(true);
  updateSeedDisplay(); // Ensure seed display is current
}

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
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
  let numSamples = Math.pow(2, state.numBits);
  // Problem type is removed, default to state.dataset for data generation
  let generator = state.dataset;
  let data = generator(numSamples, state.noise / 100);
  trainData = data;
  testData = data;
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(testData);
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

initTutorial();
makeGUI();
// state.seed is initialized to "0" by state.ts on first load if not in hash.
// generateData() will use this seed.
generateData();
reset(true, true); // true for onStartup, true for hardcodeWeights (because seed is "0")
hideControls();

function initCollapsibleSections() {
  // Initialize collapse buttons for each section
  const sections = [
    { selector: '.instructions.column', stateKey: 'instructionsCollapsed' },
    { selector: '.ai-safety.column', stateKey: 'aiSafetyCollapsed' },
    { selector: '.features.column', stateKey: 'featuresCollapsed' },
    { selector: '.output.column', stateKey: 'outputCollapsed' },
    { selector: '.code.column', stateKey: 'codeCollapsed' }
  ];

  sections.forEach(section => {
    const column = d3.select(section.selector);
    const collapseButton = column.select('.collapse-button');
    const headerWrapper = column.select('.header-wrapper');

    // Add click handler to the header
    headerWrapper.on('click', () => {
      toggleSection(section.selector, section.stateKey);
    });

    // Apply initial collapsed state
    if (state[section.stateKey]) {
      column.classed('collapsed', true);
      collapseButton.style('transform', 'rotate(-90deg)');
    }
  });
}

function toggleSection(selector: string, stateKey: string) {
  const column = d3.select(selector);
  const collapseButton = column.select('.collapse-button');
  const isCollapsed = column.classed('collapsed');

  // Update state
  state[stateKey] = !isCollapsed;

  // Toggle collapsed class
  column.classed('collapsed', !isCollapsed);

  if (!isCollapsed) {
    collapseButton.style('transform', 'rotate(90deg)');
  } else {
    collapseButton.style('transform', 'rotate(0deg)');
  }
}
