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
import {ActivationFunction, Activations} from "./activation";
import {Range, addRange, multiplyRange, activationRange, BIT_RANGES} from "./range";

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  /** The sum of the inputs * weights + bias. */
  totalInput: number;
  /** relu(totalInput), or whatever is the activation function */
  output: number;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /**
   * The error derivative with respect to this node's total input.
   * Also known as delta.
   */
  delta = 0;
  /**
   * Accumulated delta for the bias since the last update.
   * This is dE/db.
   */
  accBiasDelta = 0;
  /**
   * Number of accumulated deltas for the bias since the last update.
   */
  numAccumulated = 0;
  /**
   * Accumulated delta for the bias since the last update, but for the
   * theoretical calculations.
   */
  theoryAccBiasDelta = 0;
  /**
   * Number of accumulated deltas for the bias since the last update, but for
   * the theoretical calculations.
   */
  numTheoryAccumulated = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;
  /** Range of possible values for the sum of the inputs * weights + bias. */
  totalInputRange: Range;
  /** Range of possible values for relu(totalInputRange), or whatever is the activation function. */
  outputRange: Range;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction) {
    this.id = id;
    this.activation = activation;
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5; // Will use seedrandom via playground.ts
  isDead = false;
  /** Accumulated delta for the weight since the last update. */
  accWeightDelta = 0;
  /** Number of accumulated deltas for the weight since the last update. */
  numAccumulated = 0;
  /**
   * Accumulated delta for the weight since the last update, but for the
   * theoretical calculations.
   */
  theoryAccWeightDelta = 0;
  /**
   * Number of accumulated deltas for the weight since the last update, but for
   * the theoretical calculations.
   */
  numTheoryAccumulated = 0;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   */
  constructor(source: Node, dest: Node) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    inputIds: string[]): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId, isOutputLayer ? outputActivation : activation);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The output node.
 */
export function forwardProp(network: Node[][], inputs: number[]): Node {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
      " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    const node = inputLayer[i];
    node.output = inputs[i];
  }

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      node.updateOutput();
    }
  }
  const outputNode = network[network.length - 1][0];
  return outputNode;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction, theory = false): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  const outputNode = network[network.length - 1][0];
  const outputValue = theory ? outputNode.outputRange[1] : outputNode.output;
  outputNode.outputDer = errorFunc.der(outputValue, target);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      node.delta = node.outputDer *
          node.activation.der(
              theory ? node.totalInputRange[1] : node.totalInput);
      if (theory) {
        node.theoryAccBiasDelta += node.delta;
        node.numTheoryAccumulated++;
      } else {
        node.accBiasDelta += node.delta;
        node.numAccumulated++;
      }
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        const sourceValue =
            theory ? link.source.outputRange[1] : link.source.output;
        const weightDelta = node.delta * sourceValue;
        if (theory) {
          link.theoryAccWeightDelta += weightDelta;
          link.numTheoryAccumulated++;
        } else {
          link.accWeightDelta += weightDelta;
          link.numAccumulated++;
        }
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        const output = node.outputs[j];
        node.outputDer += output.weight * output.dest.delta;
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    safetyImportance: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      let normalAccBiasDelta = node.accBiasDelta;
      let numNormalAccumulated = node.numAccumulated;
      let theoryAccBiasDelta = node.theoryAccBiasDelta;
      let numTheoryAccumulated = node.numTheoryAccumulated;

      if (numNormalAccumulated > 0 || numTheoryAccumulated > 0) {
        let normalDelta = numNormalAccumulated > 0 ?
            normalAccBiasDelta / numNormalAccumulated :
            0;
        let theoryDelta = numTheoryAccumulated > 0 ?
            theoryAccBiasDelta / numTheoryAccumulated :
            0;

        let totalDelta = (1 - safetyImportance) * normalDelta +
            safetyImportance * theoryDelta;
        node.bias -= learningRate * totalDelta;

        node.accBiasDelta = 0;
        node.numAccumulated = 0;
        node.theoryAccBiasDelta = 0;
        node.numTheoryAccumulated = 0;
      }

      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }

        let normalAccWeightDelta = link.accWeightDelta;
        let numNormalAccumulated = link.numAccumulated;
        let theoryAccWeightDelta = link.theoryAccWeightDelta;
        let numTheoryAccumulated = link.numTheoryAccumulated;

        if (numNormalAccumulated > 0 || numTheoryAccumulated > 0) {
          let normalDelta = numNormalAccumulated > 0 ?
              normalAccWeightDelta / numNormalAccumulated :
              0;
          let theoryDelta = numTheoryAccumulated > 0 ?
              theoryAccWeightDelta / numTheoryAccumulated :
              0;

          let totalDelta = (1 - safetyImportance) * normalDelta +
              safetyImportance * theoryDelta;
          link.weight = link.weight - learningRate * totalDelta;

          link.accWeightDelta = 0;
          link.numAccumulated = 0;
          link.theoryAccWeightDelta = 0;
          link.numTheoryAccumulated = 0;
        }
      }
    }
  }
  updateNodeRanges(network, BIT_RANGES);
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}

/**
 * Updates the ranges of all nodes in the network.
 *
 * @param network The neural network.
 * @param activationFunction The activation function used in the network.
 * @param inputRanges A map from input node id to its range.
 */
export function updateNodeRanges(network: Node[][],
    inputRanges: Map<string, Range>): void {
  // Set the initial ranges in the input layer.
  let inputLayer = network[0];
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.outputRange = inputRanges.get(node.id) || [0.0, 1.0];
  }

  // Propagate the ranges for hidden and output layers.
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let currentRange: Range = [node.bias, node.bias];

      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let inputNode = link.source;
        let weightedRange = multiplyRange(link.weight, inputNode.outputRange);
        currentRange = addRange(currentRange, weightedRange);
      }
      node.totalInputRange = currentRange;
      node.outputRange = activationRange(node.activation, currentRange);
    }
  }
}
