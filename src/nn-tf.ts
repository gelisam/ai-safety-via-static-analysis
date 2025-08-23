/* Copyright 2020 Google LLC. All Rights Reserved.

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
import { ActivationFunction, Activations } from "./activation";
import { Range, addRange, multiplyRange, activationRange, BIT_RANGES } from "./range";
import * as tf from "@tensorflow/tfjs";
import {Example2D} from "./dataset";

// Keep Node and Link for type compatibility with playground.ts
export class Node {
    id: string;
    inputLinks: Link[] = [];
    bias: number = 0.1;
    outputs: Link[] = [];
    totalInput: number;
    output: number;
    outputDer = 0;
    inputDer = 0;
    accInputDer = 0;
    numAccumulatedDers = 0;
    theoryAccInputDer = 0;
    numTheoryAccumulatedDers = 0;
    activation: ActivationFunction;
    range: Range;

    constructor(id: string, activation: ActivationFunction) {
        this.id = id;
        this.activation = activation;
    }

    updateOutput(): number {
        return 0; // This will be replaced by tensorflow.js
    }
}

export class Link {
    id: string;
    source: Node;
    dest: Node;
    weight: number = Math.random() - 0.5;
    isDead = false;
    errorDer = 0;
    accErrorDer = 0;
    numAccumulatedDers = 0;
    theoryAccErrorDer = 0;
    numTheoryAccumulatedDers = 0;

    constructor(source: Node, dest: Node) {
        this.id = source.id + "-" + dest.id;
        this.source = source;
        this.dest = dest;
    }
}

export interface ErrorFunction {
  error: (output: tf.Tensor, target: tf.Tensor) => tf.Tensor;
  der: (output: tf.Tensor, target: tf.Tensor) => tf.Tensor;
}

export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: tf.Tensor, target: tf.Tensor) =>
        tf.mean(tf.pow(tf.sub(output, target), 2)).asScalar(),
    der: (output: tf.Tensor, target: tf.Tensor) => tf.sub(output, target)
  };
}

function getTfActivation(activation: ActivationFunction): string {
    if (activation.name === "relu") {
        return "relu";
    }
    if (activation.name === "tanh") {
        return "tanh";
    }
    if (activation.name === "sigmoid") {
        return "sigmoid";
    }
    if (activation.name === "linear") {
        return "linear";
    }
    throw new Error(`Unknown activation function ${activation.name}`);
}

export function buildNetwork(
    networkShape: number[],
    activation: ActivationFunction,
    outputActivation: ActivationFunction,
    inputIds: string[]
): {model: tf.Sequential, network: Node[][]} {
    const model = tf.sequential();
    const network: Node[][] = [];
    const numLayers = networkShape.length;

    // Input layer
    const inputLayer: Node[] = [];
    const inputLayerShape = networkShape[0];
    for (let i = 0; i < inputLayerShape; i++) {
        const nodeId = inputIds[i];
        const node = new Node(nodeId, null);
        inputLayer.push(node);
    }
    network.push(inputLayer);


    // Hidden layers and output layer
    for (let i = 1; i < networkShape.length; i++) {
        const numNodes = networkShape[i];
        const prevNumNodes = networkShape[i - 1];
        const layerActivation = i === networkShape.length - 1 ? outputActivation : activation;

        const denseLayer = tf.layers.dense({
            units: numNodes,
            inputShape: [prevNumNodes],
            activation: getTfActivation(layerActivation) as any,
        });
        model.add(denseLayer);

        const currentLayer: Node[] = [];
        let id = (i-1) * 10;
        for (let j = 0; j < numNodes; j++) {
            const nodeId = id.toString();
            id++;
            const node = new Node(nodeId, layerActivation);
            currentLayer.push(node);
        }
        network.push(currentLayer);
    }

    // This is a hack to make the old code work.
    // The old code expects the network to be built with links.
    // We will create the links here, but they will not be used by tensorflow.js
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
        let currentLayer = network[layerIdx];
        let prevLayer = network[layerIdx - 1];
        for (let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            for (let j = 0; j < prevLayer.length; j++) {
                let prevNode = prevLayer[j];
                let link = new Link(prevNode, node);
                prevNode.outputs.push(link);
                node.inputLinks.push(link);
            }
        }
    }

    return {model, network};
}

export function setWeights(model: tf.Sequential, network: Node[][]) {
    for (let i = 1; i < network.length; i++) {
        const layer = network[i];
        const prevLayer = network[i-1];
        const layerIndex = i - 1;
        const layerWeighs: tf.Tensor[] = [];

        const kernel = tf.tensor2d(
            Array.from({length: prevLayer.length}, (_, prevNodeIndex) =>
                Array.from({length: layer.length}, (_, nodeIndex) =>
                    layer[nodeIndex].inputLinks[prevNodeIndex].weight
                )
            )
        );
        layerWeighs.push(kernel);

        const bias = tf.tensor1d(
            Array.from({length: layer.length}, (_, nodeIndex) =>
                layer[nodeIndex].bias
            )
        );
        layerWeighs.push(bias);
        model.layers[layerIndex].setWeights(layerWeighs);
    }
}

export async function getWeights(model: tf.Sequential, network: Node[][]) {
    for (let i = 1; i < network.length; i++) {
        const layer = network[i];
        const prevLayer = network[i-1];
        const layerIndex = i - 1;
        const weights = model.layers[layerIndex].getWeights();
        const kernel = await weights[0].array() as number[][];
        const bias = await weights[1].array() as number[];

        for(let k=0; k < layer.length; k++) {
            layer[k].bias = bias[k];
            for (let j=0; j < prevLayer.length; j++) {
                layer[k].inputLinks[j].weight = kernel[j][k];
            }
        }
    }
}

export function forwardProp(model: tf.Sequential, network: Node[][], inputs: number[]): number {
    const inputTensor = tf.tensor2d([inputs], [1, inputs.length]);
    let currentTensor = inputTensor;

    for (let i = 0; i < model.layers.length; i++) {
        const layer = model.layers[i];
        currentTensor = layer.apply(currentTensor) as tf.Tensor;
        const output = currentTensor.dataSync();
        const currentLayer = network[i + 1];
        for (let j = 0; j < currentLayer.length; j++) {
            currentLayer[j].output = output[j];
        }
    }
    return currentTensor.dataSync()[0];
}

export async function oneStep(model: tf.Sequential, network: Node[][], trainData: Example2D[], learningRate: number, constructInput: (x: number, y: number) => number[]) {
    const inputs = trainData.map(d => constructInput(d.x, d.y));
    const xs = tf.tensor2d(inputs, [trainData.length, inputs[0].length]);
    const ys = tf.tensor2d(trainData.map(d => [d.label]), [trainData.length, 1]);

    const optimizer = tf.train.sgd(learningRate);

    const loss = (pred: tf.Tensor, label: tf.Tensor) => tf.mean(tf.square(tf.sub(pred, label))).mul(0.5).asScalar();

    await optimizer.minimize(() => loss(model.apply(xs) as tf.Tensor, ys));

    await getWeights(model, network);
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
    node.range = inputRanges.get(node.id) || [0.0, 1.0];
  }

  // Propagate the ranges for hidden and output layers.
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let currentRange: Range = [node.bias as number, node.bias as number];

      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let inputNode = link.source;
        let weightedRange = multiplyRange(link.weight as number, inputNode.range);
        currentRange = addRange(currentRange, weightedRange);
      }
      node.range = activationRange(node.activation, currentRange);
    }
  }
}
