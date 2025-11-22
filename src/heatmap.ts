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

import {Example2D} from "./dataset";
import * as d3 from 'd3';
import * as nn from "./nn";

// Interface definition moved to module level
interface PredictiveNetworkWrapper {
  predict: (point: Example2D) => number;
}

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

/** Number of different shades (colors) when drawing a gradient heatmap */
const NUM_SHADES = 30;

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class HeatMap {
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false
  };
  private xScale;
  private yScale;
  private numSamples: number;
  private color;
  private defaultColor;
  private canvas;
  private svg;
  private topRowRect: d3.Selection<SVGElement>;

  constructor(
      width: number, numSamples: number, xDomain: [number, number],
      yDomain: [number, number], container,
      userSettings?: HeatMapSettings) {
    this.numSamples = numSamples;
    let height = width;
    let padding = userSettings.showAxes ? 20 : 0;

    if (userSettings != null) {
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3.scale.linear()
      .domain(xDomain)
      .range([0, width - 2 * padding]);

    this.yScale = d3.scale.linear()
      .domain(yDomain)
      .range([height - 2 * padding, 0]);

    let tmpScale = d3.scale.linear<string, number>()
        .domain([0, .5, 1])
        .range(["#f59322", "#e8eaeb", "#0877bd"])
        .clamp(true);
    let colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(a => {
      return tmpScale(a);
    });
    this.color = d3.scale.quantize()
                     .domain([-1, 1])
                     .range(colors);
    this.defaultColor = this.color;

    container = container.append("div")
      .style({
        width: `${width}px`,
        height: `${height}px`,
        position: "relative",
        top: `-${padding}px`,
        left: `-${padding}px`
      });
    this.canvas = container.append("canvas")
      .attr("width", numSamples)
      .attr("height", numSamples)
      .style("width", (width - 2 * padding) + "px")
      .style("height", (height - 2 * padding) + "px")
      .style("position", "absolute")
      .style("top", `${padding}px`)
      .style("left", `${padding}px`);

    if (!this.settings.noSvg) {
      this.svg = container.append("svg").attr({
          "width": width,
          "height": height
      }).style({
        "position": "absolute",
        "left": "0",
        "top": "0"
      }).append("g")
        .attr("transform", `translate(${padding},${padding})`);

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    if (this.settings.showAxes) {
      let xAxis = d3.svg.axis()
        .scale(this.xScale)
        .orient("bottom");
      let yAxis = d3.svg.axis()
        .scale(this.yScale)
        .orient("right");
      this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(0,${height - 2 * padding})`)
        .call(xAxis);
      this.svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (width - 2 * padding) + ",0)")
        .call(yAxis);
    }
  }

  updateTestPoints(points: Example2D[], network?: any): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select("g.test"), points, network);
  }

  updatePoints(points: Example2D[], network?: any): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select("g.train"), points, network);
  }

  updateBackground(data: number[][]): void {
    let dx = data[0].length;
    let dy = data.length;
    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error(
          "The provided data matrix must be of size " +
          "numSamples X numSamples");
    }
    let context = (this.canvas.node() as HTMLCanvasElement).getContext("2d");
    let image = context.createImageData(dx, dy);
    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        let c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  updateColorScale(domain: [number, number, number]): void {
    this.color.domain(domain);
  }

  restoreDefaultColorScale(): void {
    this.color = this.defaultColor;
  }

  drawTopRowHighlight(allPoints: Example2D[], network?: PredictiveNetworkWrapper): void {
    if (this.settings.noSvg || !this.svg) {
      return;
    }
    if (this.topRowRect) {
      this.topRowRect.remove();
      this.topRowRect = null;
    }
    if (!allPoints || allPoints.length === 0) {
      return;
    }
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    const visiblePoints = allPoints.filter(p => {
      return p.x >= xDomain[0] && p.x <= xDomain[1] &&
             p.y >= yDomain[0] && p.y <= yDomain[1];
    });
    if (visiblePoints.length === 0) {
      return;
    }
    let maxYValue = -Infinity;
    visiblePoints.forEach(p => {
      if (p.y > maxYValue) {
        maxYValue = p.y;
      }
    });
    const epsilon = 0.0001;
    const topRowPoints = visiblePoints.filter(p => Math.abs(p.y - maxYValue) < epsilon);
    if (topRowPoints.length === 0) {
      return;
    }
    let minX = Infinity, maxX = -Infinity;
    topRowPoints.forEach(p => {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
    });
    const pointYValue = maxYValue;
    const padding = 0.3;
    const rectX = this.xScale(minX - padding);
    const rectY = this.yScale(pointYValue + padding);
    const rectWidth = this.xScale(maxX + padding) - this.xScale(minX - padding);
    const rectHeight = this.yScale(pointYValue - padding) - this.yScale(pointYValue + padding);

    if (rectWidth <= 0 || rectHeight <= 0) return;

    let rectStrokeColor = "green";
    if (network && topRowPoints.length > 0) {
      for (const point of topRowPoints) {
        if (point.bits && point.bits.length >= 4 &&
            point.bits[0] && point.bits[1] && point.bits[2] && point.bits[3]) {
          if (network.hasOwnProperty('predict')) {
            const prediction = (network as any).predict(point);
            if (prediction !== point.label) {
              rectStrokeColor = "red";
              break;
            }
          }
        }
      }
    }
    const inset = 2;
    let finalRectX = rectX + inset;
    let finalRectY = rectY + inset;
    let finalRectWidth = rectWidth - (2 * inset);
    let finalRectHeight = rectHeight - (2 * inset);

    if (finalRectWidth <= 0 || finalRectHeight <= 0) return;

    this.topRowRect = this.svg.append("rect")
      .attr("x", finalRectX)
      .attr("y", finalRectY)
      .attr("width", finalRectWidth)
      .attr("height", finalRectHeight)
      .attr("fill", "none")
      .attr("stroke", rectStrokeColor)
      .attr("stroke-width", 3);
  }

  private bitCount(bits: boolean[]): number {
    let count = 0;
    for (let i = 0; i < bits.length; i++) {
      if (bits[i]) {
        count++;
      }
    }
    return count;
  }

  private updateCircles(container, points: Example2D[], network?: any) {
    let xDomain = this.xScale.domain();
    let yDomain = this.yScale.domain();
    points = points.filter(p => {
      return p.x >= xDomain[0] && p.x <= xDomain[1]
        && p.y >= yDomain[0] && p.y <= yDomain[1];
    });
    container.selectAll("g.point").remove();
    let pointGroups = container.selectAll("g.point")
      .data(points)
      .enter()
      .append("g")
      .attr("class", "point")
      .attr("transform", (d: Example2D) =>
        `translate(${this.xScale(d.x)},${this.yScale(d.y)})`);
    pointGroups.append("circle")
      .attr("r", 12)
      .style("fill", d => this.color(d.label))
      .style("stroke", "black")
      .style("stroke-width", "1px");
    pointGroups.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-0.6em")
      .style("font-size", "7px")
      .style("font-family", "monospace")
      .style("fill", "white")
      .text((d: Example2D) => {
        if (!d.bits) return "";
        let firstHalf = d.bits.slice(0, 4).map(b => b ? "1" : "0").join("");
        return firstHalf;
      });
    pointGroups.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .style("font-size", "7px")
      .style("font-family", "monospace")
      .style("fill", "white")
      .text((d: Example2D) => {
        if (!d.bits) return "";
        let secondHalf = d.bits.slice(4).map(b => b ? "1" : "0").join("");
        return secondHalf;
      });
    pointGroups.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.1em")
      .style("font-size", "7px")
      .style("font-family", "monospace")
      .style("fill", "white")
      .text((d: Example2D) => {
        if (!d.bits) return "";
        return "" + this.bitCount(d.bits);
      });
    if (network) {
      pointGroups.append("circle")
        .attr("r", 12)
        .style("fill", "none")
        .style("stroke", "red")
        .style("stroke-width", "2px")
        .style("opacity", d => network.predict(d) === d.label ? 0 : 1);
    }
  }
}

export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) {
    throw new Error("The provided matrix must be a square matrix");
  }
  if (matrix.length % factor !== 0) {
    throw new Error("The width/height of the matrix must be divisible by " +
        "the reduction factor");
  }
  let result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      for (let k = 0; k < factor; k++) {
        for (let l = 0; l < factor; l++) {
          avg += matrix[i + k][j + l];
        }
      }
      avg /= (factor * factor);
      result[i / factor][j / factor] = avg;
    }
  }
  return result;
}
