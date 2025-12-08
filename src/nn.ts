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
 * (pre-activation, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  bias = 0.1;
  /** List of input links. */
  inputLinks: Link[] = [];
  /** List of output links. */
  outputLinks: Link[] = [];
  /** Activation function that takes pre-activation and returns node's output */
  activation: ActivationFunction;

  /** The sum of the inputs * weights + bias. */
  preActivation: number;
  /** Range of possible values for the sum of the inputs * weights + bias. */
  preActivationRange: Range;

  /** relu(preActivation), or whatever is the activation function */
  output: number;
  /** Derivative of the output with respect to the tweaked input. */
  outputDeriv: number;
  /** Range of possible values for relu(preActivationRange), or whatever is the activation function. */
  outputRange: Range;

  /** To minimize the squared error, we would like to change the output in this direction. */
  desiredDeltaOutput: number;
  /** To minimize the squared error on safety-in-theory, we would like to change
   * each component of 'outputRange' in this direction.
   */
  desiredDeltaOutputComponents: [number, number];

  /** Sum of 'desiredDeltaPreActivation' for many samples. */
  accDesiredDeltaPreActivation = 0;
  /** Component-wise sum of 'desiredDeltaPreActivationComponents' for many samples. */
  accDesiredDeltaPreActivationComponents: [number, number] = [0, 0];

  /** Number of samples that have contributed to 'accDesiredDeltaPreActivation'. */
  numDesiredDeltaPreActivations = 0;
  /** Number of samples that have contributed to 'accDesiredDeltaPreActivationComponents'. */
  numDesiredDeltaPreActivationComponents = 0;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction) {
    this.id = id;
    this.activation = activation;
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores pre-activation into the node.
    this.preActivation = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.preActivation += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.preActivation);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  /** f(output - target) */
  error: (output: number, target: number) => number;
  /** f'(output - target) */
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

  /** To minimize the squared error, we would like to change the input in this direction. */
  desiredDeltaInput: number;
  /** To minimize the squared error on safety-in-theory, we would like to change
   * each component of the input range in this direction.
   */
  desiredDeltaInputComponents: [number, number];

  /** Sum of 'desiredDeltaWeight' for many downstream nodes and many samples. */
  accDesiredDeltaWeight = 0;
  /** Component-wise sum of 'desiredDeltaWeightComponents' for many downstream nodes and many samples. */
  accDesiredDeltaWeightComponents: [number, number] = [0, 0];

  /** Number of samples that have contributed to 'accDesiredDeltaWeight'. */
  numDesiredDeltaWeights = 0;
  /** Number of samples that have contributed to 'accDesiredDeltaWeightComponents'. */
  numDesiredDeltaWeightComponents = 0;

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
          prevNode.outputLinks.push(link);
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
 * pre-activation and output of each node in the network.
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
 * Runs a forward propagation of the derivatives through the provided
 * network. This method modifies the internal state of the network - the
 * pre-activation and output derivatives of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The output node.
 */
export function forwardPropDeriv(network: Node[][],
    tweak: Map<string, number>): Map<string, number> {
  for (let layerIdx = 0; layerIdx < network.length; layerIdx++) {
    for (const node of network[layerIdx]) {
      let propagatedDeriv = 0;
      if (layerIdx > 0) {
        let preActivationDeriv = 0;
        for (const link of node.inputLinks) {
          preActivationDeriv += link.weight * link.source.outputDeriv;
        }
        propagatedDeriv = node.activation.der(node.preActivation) * preActivationDeriv;
      }
      node.outputDeriv = (tweak.has(node.id) ? tweak.get(node.id) : 0) + propagatedDeriv;
    }
  }

  let allDerivs = new Map<string, number>();
  forEachNode(network, false, node => allDerivs.set(node.id, node.outputDeriv));
  return allDerivs;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  const outputNode = network[network.length - 1][0];
  const outputValue = outputNode.output;
  const desiredDeltaSquaredError = -1; // We want to reduce the error.
  const squaredErrorWrtOutput = errorFunc.der(outputValue, target);
  outputNode.desiredDeltaOutput = desiredDeltaSquaredError * squaredErrorWrtOutput;

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    // At this point we know how much we want to change the output of each node
    // in this layer. Use it to calculate how much we want to change the
    // pre-activation of each node.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      const outputWrtPreActivation = node.activation.der(node.preActivation);
      const desiredDeltaPreActivation = node.desiredDeltaOutput * outputWrtPreActivation;
      node.accDesiredDeltaPreActivation += desiredDeltaPreActivation;
      node.numDesiredDeltaPreActivations++;

      // Now that we know how much we want to change the pre-activation the node,
      // we can calculate how much we want to change each weight and each input.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        // contribution = weight * input
        const input = link.source.output;
        const desiredDeltaContribution = desiredDeltaPreActivation;
        const contributionWrtWeight = input;
        const contributionWrtInput = link.weight;
        const desiredDeltaWeight = desiredDeltaContribution * contributionWrtWeight;
        link.desiredDeltaInput = desiredDeltaContribution * contributionWrtInput;
        link.accDesiredDeltaWeight += desiredDeltaWeight;
        link.numDesiredDeltaWeights++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      // Sum up the desired delta inputs from all downstream links in order to
      // compute this node's desired delta output.
      node.desiredDeltaOutput = 0;
      for (let j = 0; j < node.outputLinks.length; j++) {
        const downstreamLink = node.outputLinks[j];
        node.desiredDeltaOutput += downstreamLink.desiredDeltaInput;
      }
    }
  }
}

/**
 * Runs a backward propagation of the derivatives through the provided
 * network.
 *
 * @param network The neural network.
 * @param tweak A map from node id to the derivative of the tweak wrt the
 *     output of that node.
 * @return A map from node id to the derivative of the tweak wrt the
 *     output of that node.
 */
export function backPropDeriv(network: Node[][],
    tweak: Map<string, number>): Map<string, number> {
  let tweakedDerivs = new Map<string, number>();
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      let desiredDeltaOutput = 0;
      if (tweak.has(node.id)) {
        desiredDeltaOutput += tweak.get(node.id);
      }
      if (tweakedDerivs.has(node.id)) {
        desiredDeltaOutput += tweakedDerivs.get(node.id);
      }

      const outputWrtPreActivation = node.activation.der(node.preActivation);
      const desiredDeltaPreActivation = desiredDeltaOutput * outputWrtPreActivation;

      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        const contributionWrtInput = link.weight;
        const desiredDeltaInput = desiredDeltaPreActivation * contributionWrtInput;
        const sourceNode = link.source;
        let sourceDeriv = tweakedDerivs.has(sourceNode.id) ? tweakedDerivs.get(sourceNode.id) : 0;
        sourceDeriv += desiredDeltaInput;
        tweakedDerivs.set(sourceNode.id, sourceDeriv);
      }
    }
  }
  return tweakedDerivs;
}

export function backPropRanges(network: Node[][], target: Range,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  const outputNode = network[network.length - 1][0];
  // const outputValue = outputNode.output;
  // const desiredDeltaSquaredError = -1;
  // const squaredErrorWrtOutput = errorFunc.der(outputValue, target);
  // outputNode.desiredDeltaOutput = desiredDeltaSquaredError * squaredErrorWrtOutput;
  const outputComponent0 = outputNode.outputRange[0];
  const outputComponent1 = outputNode.outputRange[1];
  const desiredDeltaSquaredError = -1; // We want to reduce the error.
  const squaredErrorWrtOutputComponent0 = errorFunc.der(outputComponent0, target[0]);
  const squaredErrorWrtOutputComponent1 = errorFunc.der(outputComponent1, target[1]);
  const desiredDeltaOutputComponent0 = desiredDeltaSquaredError * squaredErrorWrtOutputComponent0;
  const desiredDeltaOutputComponent1 = desiredDeltaSquaredError * squaredErrorWrtOutputComponent1;
  outputNode.desiredDeltaOutputComponents = [desiredDeltaOutputComponent0, desiredDeltaOutputComponent1];

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    // At this point we know how much we want to change the output of each node
    // in this layer. Use it to calculate how much we want to change the
    // pre-activation of each node.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      // const outputWrtPreActivation = node.activation.der(node.preActivation);
      // const desiredDeltaPreActivation = node.desiredDeltaOutput * outputWrtPreActivation;
      // node.accDesiredDeltaPreActivation += desiredDeltaPreActivation;
      // node.numDesiredDeltaPreActivations++;
      const outputWrtPreActivation0 = node.activation.der(node.preActivationRange[0]);
      const outputWrtPreActivation1 = node.activation.der(node.preActivationRange[1]);
      const desiredDeltaPreActivation0 = node.desiredDeltaOutputComponents[0] * outputWrtPreActivation0;
      const desiredDeltaPreActivation1 = node.desiredDeltaOutputComponents[1] * outputWrtPreActivation1;
      const accDesiredPreActivation0 = node.accDesiredDeltaPreActivationComponents[0] + desiredDeltaPreActivation0;
      const accDesiredPreActivation1 = node.accDesiredDeltaPreActivationComponents[1] + desiredDeltaPreActivation1;
      node.accDesiredDeltaPreActivationComponents = [accDesiredPreActivation0, accDesiredPreActivation1];
      node.numDesiredDeltaPreActivationComponents++;

      // Now that we know how much we want to change the pre-activation the node,
      // we can calculate how much we want to change each weight and each input.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        // const input = link.source.output;
        // const desiredDeltaContribution = desiredDeltaPreActivation;
        // const contributionWrtWeight = input;
        // const contributionWrtInput = link.weight;
        // const desiredDeltaWeight = desiredDeltaContribution * contributionWrtWeight;
        // link.desiredDeltaInput = desiredDeltaContribution * contributionWrtInput;
        // link.accDesiredDeltaWeight += desiredDeltaWeight;
        // link.numDesiredDeltaWeights++;
        const input0 = link.source.outputRange[0];
        const input1 = link.source.outputRange[1];
        const desiredDeltaContribution0 = desiredDeltaPreActivation0;
        const desiredDeltaContribution1 = desiredDeltaPreActivation1;
        const contributionWrtWeight0 = input0;
        const contributionWrtWeight1 = input1;
        const contributionWrtInput = link.weight;
        if (link.weight > 0) {
          const desiredDeltaWeight0 = desiredDeltaContribution0 * contributionWrtWeight0;
          const desiredDeltaWeight1 = desiredDeltaContribution1 * contributionWrtWeight1;
          const desiredDeltaInput0 = desiredDeltaContribution0 * contributionWrtInput;
          const desiredDeltaInput1 = desiredDeltaContribution1 * contributionWrtInput;
          link.desiredDeltaInputComponents = [desiredDeltaInput0, desiredDeltaInput1];
          link.accDesiredDeltaWeightComponents[0] += desiredDeltaWeight0;
          link.accDesiredDeltaWeightComponents[1] += desiredDeltaWeight1;
          link.numDesiredDeltaWeightComponents++;
        } else {
          // the weight is negative, so min and max swap roles:
          //   -1 * [-2, -1] = [1, 2]
          // if we want to increase the upper bound 2 of the output, that means
          // we need to _decrease_ the _lower_ bound -2 of the input.
          const desiredDeltaWeight0 = desiredDeltaContribution1/*!*/ * contributionWrtWeight0;
          const desiredDeltaWeight1 = desiredDeltaContribution0/*!*/ * contributionWrtWeight1;
          const desiredDeltaInput0 = desiredDeltaContribution1/*!*/ * contributionWrtInput;
          const desiredDeltaInput1 = desiredDeltaContribution0/*!*/ * contributionWrtInput;
          link.desiredDeltaInputComponents = [desiredDeltaInput0, desiredDeltaInput1];
          link.accDesiredDeltaWeightComponents[0] += desiredDeltaWeight0;
          link.accDesiredDeltaWeightComponents[1] += desiredDeltaWeight1;
          link.numDesiredDeltaWeightComponents++;
        }
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      // Sum up the desired delta inputs from all downstream links in order to
      // compute this node's desired delta output.
      // node.desiredDeltaOutput = 0;
      node.desiredDeltaOutputComponents = [0, 0];
      for (let j = 0; j < node.outputLinks.length; j++) {
        const downstreamLink = node.outputLinks[j];
        // node.desiredDeltaOutput += downstreamLink.desiredDeltaInput;
        node.desiredDeltaOutputComponents[0] += downstreamLink.desiredDeltaInputComponents[0];
        node.desiredDeltaOutputComponents[1] += downstreamLink.desiredDeltaInputComponents[1];
      }
    }
  }
}

function safeDivide(numerator: number, denominator: number): number {
  if (denominator === 0) {
    return 0;
  }
  return numerator / denominator;
}

/**
 * Update the weights of the network using the accumulated desired deltas.
 */
export function updateWeights(network: Node[][], learningRate: number,
    safetyImportance: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      const accDesiredDeltaBias = node.accDesiredDeltaPreActivation;
      const numDesiredDeltaBias = node.numDesiredDeltaPreActivations;
      const practicalDesiredDeltaBias = safeDivide(accDesiredDeltaBias, numDesiredDeltaBias);

      const accDesiredDeltaBias0 = node.accDesiredDeltaPreActivationComponents[0];
      const accDesiredDeltaBias1 = node.accDesiredDeltaPreActivationComponents[1];
      const desiredDeltaBias0 = safeDivide(accDesiredDeltaBias0, node.numDesiredDeltaPreActivationComponents);
      const desiredDeltaBias1 = safeDivide(accDesiredDeltaBias1, node.numDesiredDeltaPreActivationComponents);
      const safetyDesiredDeltaBias = desiredDeltaBias0 + desiredDeltaBias1; // divide by 2?

      const totalDesiredDeltaBias = (1 - safetyImportance) * practicalDesiredDeltaBias + safetyImportance * safetyDesiredDeltaBias;
      node.bias += learningRate * totalDesiredDeltaBias;

      node.accDesiredDeltaPreActivation = 0;
      node.numDesiredDeltaPreActivations = 0;
      node.accDesiredDeltaPreActivationComponents = [0, 0];
      node.numDesiredDeltaPreActivationComponents = 0;

      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }

        const practicalDesiredDeltaWeight = safeDivide(link.accDesiredDeltaWeight, link.numDesiredDeltaWeights);

        const theoryDesiredDeltaWeight0 = safeDivide(link.accDesiredDeltaWeightComponents[0], link.numDesiredDeltaWeightComponents);
        const theoryDesiredDeltaWeight1 = safeDivide(link.accDesiredDeltaWeightComponents[1], link.numDesiredDeltaWeightComponents);
        const safetyDesiredDeltaWeight = theoryDesiredDeltaWeight0 + theoryDesiredDeltaWeight1; // divide by 2?

        const totalDesiredDeltaWeight = (1 - safetyImportance) * practicalDesiredDeltaWeight + safetyImportance * safetyDesiredDeltaWeight;
        link.weight += learningRate * totalDesiredDeltaWeight;

        link.accDesiredDeltaWeight = 0;
        link.numDesiredDeltaWeights = 0;
        link.accDesiredDeltaWeightComponents = [0, 0];
        link.numDesiredDeltaWeightComponents = 0;
      }
    }
  }
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
export function forwardPropRanges(network: Node[][],
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
      node.preActivationRange = currentRange;
      node.outputRange = activationRange(node.activation, currentRange);
    }
  }
}
