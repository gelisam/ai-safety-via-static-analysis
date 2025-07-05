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
import * as dataset from "./dataset";

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";

/** A map between names and activation functions. */
export let activations: {[key: string]: nn.ActivationFunction} = {
  "relu": nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  "sigmoid": nn.Activations.SIGMOID,
  "linear": nn.Activations.LINEAR
};

/** A map between names and regularization functions. */
export let regularizations: {[key: string]: nn.RegularizationFunction} = {
  "none": null,
  "L1": nn.RegularizationFunction.L1,
  "L2": nn.RegularizationFunction.L2
};

/** A map between dataset names and functions that generate classification data. */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "parity": dataset.classifyParityData,
};

/** A map between dataset names and functions that generate regression data. */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// Add the GUI state.
export class State {

  private static PROPS: Property[] = [
    {name: "activation", type: Type.OBJECT, keyMap: activations},
    {name: "regularization", type: Type.OBJECT, keyMap: regularizations},
    {name: "numBits", type: Type.NUMBER},
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "regularizationRate", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "bit0", type: Type.BOOLEAN},
    {name: "bit1", type: Type.BOOLEAN},
    {name: "bit2", type: Type.BOOLEAN},
    {name: "bit3", type: Type.BOOLEAN},
    {name: "bit4", type: Type.BOOLEAN},
    {name: "bit5", type: Type.BOOLEAN},
    {name: "bit6", type: Type.BOOLEAN},
    {name: "bit7", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN},
    {name: "dataCollapsed", type: Type.BOOLEAN},
    {name: "featuresCollapsed", type: Type.BOOLEAN},
    {name: "outputCollapsed", type: Type.BOOLEAN}
  ];

  [key: string]: any;
  learningRate = 0.03;
  regularizationRate = 0;
  numBits = 8;
  batchSize = 10;
  discretize = false;
  tutorial: string = null;
  percTrainData = 50;
  activation = nn.Activations.RELU;
  regularization: nn.RegularizationFunction = null;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 2;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [8, 8];
  x = false;
  y = false;
  bit0 = true;
  bit1 = true;
  bit2 = true;
  bit3 = true;
  bit4 = true;
  bit5 = true;
  bit6 = true;
  bit7 = true;
  dataset: dataset.DataGenerator = dataset.classifyParityData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed: string;
  dataCollapsed = false;
  featuresCollapsed = false;
  outputCollapsed = false;

  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(): State {
    let map: {[key: string]: string} = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    let state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            // The + operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });
    state.numHiddenLayers = state.networkShape.length;
    if (state.seed == null) {
      // Start with seed "0.00000" for hardcoded weights by default
      state.seed = "0.00000";
    }
    Math.seedrandom(state.seed);
    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Do nothing, to prevent state persistence
  }

  /** Returns all the hidden properties. */
  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}
