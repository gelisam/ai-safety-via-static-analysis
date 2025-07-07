import {Activations, ActivationFunction} from "./activation";

/**
 * A tuple of two numbers, representing a range [min, max].
 */
export type Range = [number, number];

/**
 * Multiplies the min and max of the range by the constant.
 * If the constant is negative, the min and max swap places.
 */
export function multiplyRange(c: number, range: Range): Range {
  let r0 = c * range[0];
  let r1 = c * range[1];
  return (c >= 0) ? [r0, r1] : [r1, r0];
}

/**
 * Adds the mins together and the maxes together.
 */
export function addRange(range1: Range, range2: Range): Range {
  return [range1[0] + range2[0], range1[1] + range2[1]];
}

/**
 * Applies ReLU to both the min and max of the range.
 */
export function reluRange(range: Range): Range {
  return [Activations.RELU.output(range[0]), Activations.RELU.output(range[1])];
}

/**
 * Applies tanh to both the min and max of the range.
 */
export function tanhRange(range: Range): Range {
  return [Activations.TANH.output(range[0]), Activations.TANH.output(range[1])];
}

/**
 * Applies sigmoid to both the min and max of the range.
 */
export function sigmoidRange(range: Range): Range {
  return [Activations.SIGMOID.output(range[0]), Activations.SIGMOID.output(range[1])];
}

/**
 * Applies the given activation function to the range.
 */
export function activationRange(
    activation: ActivationFunction, range: Range): Range {
  if (activation === Activations.RELU) {
    return reluRange(range);
  } else if (activation === Activations.TANH) {
    return tanhRange(range);
  } else if (activation === Activations.SIGMOID) {
    return sigmoidRange(range);
  } else if (activation === Activations.LINEAR) {
    return range; // Linear activation doesn't change the range
  } else {
    // Should not happen with existing activation functions
    console.error("Unknown activation function for range calculation");
    return range;
  }
}

/**
 * Input ranges for bits 0 through 7, setting them to [-1.0, 1.0].
 */
export const BIT_RANGES: Map<string, Range> = new Map([
  ["bit0", [-1.0, 1.0] as Range],
  ["bit1", [-1.0, 1.0] as Range],
  ["bit2", [-1.0, 1.0] as Range],
  ["bit3", [-1.0, 1.0] as Range],
  ["bit4", [-1.0, 1.0] as Range],
  ["bit5", [-1.0, 1.0] as Range],
  ["bit6", [-1.0, 1.0] as Range],
  ["bit7", [-1.0, 1.0] as Range],
]);
