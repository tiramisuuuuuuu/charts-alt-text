import {
  ExponentialRegression,
  PolynomialRegression,
  SimpleLinearRegression,
} from 'ml-regression';

import { IsolationForest } from 'isolation-forest'

export type DataPoint = {
  x: number;
  y: number;
};
const MAX_SAMPLE_SIZE = 2000;
const MAX_POLYNOMIAL_DEGREE = 5;

// Use a max of 2000 data points to determine shape
function downsample(data: DataPoint[], maxSize: number): DataPoint[] {
  if (data.length <= maxSize) {
    return data;
  }

  const step = Math.ceil(data.length / maxSize);

  return data.filter((_, index) => index % step === 0);
}

function countTurningPnts(
  polyModel: { predict(x: number): number },
  testX: number[],
) {
  const xs = [];
  const ys = [];

  const samples = testX.length;
  const minX = Math.min(...testX);
  const maxX = Math.max(...testX);

  for (let i = 0; i < samples; i += 1) {
    const x = minX + ((maxX - minX) * i) / (samples - 1);
    xs.push(x);
    ys.push(polyModel.predict(x));
  }
  const slopes = [];

  for (let i = 1; i < ys.length; i += 1) {
    slopes.push(ys[i]! - ys[i - 1]!);
  }

  let turningPoints = 0;

  for (let i = 1; i < slopes.length; i += 1) {
    if (
      (slopes[i - 1]! > 0 && slopes[i]! < 0) ||
      (slopes[i - 1]! < 0 && slopes[i]! > 0)
    ) {
      turningPoints += 1;
    }
  }

  return turningPoints;
}

function std(array: number[]) {
  const n = array.length;
  if (!n) return 0;
  
  const mean = array.reduce((a, b) => a + b) / n;
  const variance = array.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
  
  return Math.sqrt(variance);
}

function r2(
  model: { predict(x: number): number },
  testX: number[],
  testY: number[],
): number {
  if (testX.length !== testY.length) return 0;

  const mean = testY.reduce((a, b) => a + b, 0) / testY.length;

  let ssRes = 0;
  let ssTot = 0;

  for (let i = 0; i < testX.length; i += 1) {
    const predicted = model.predict(testX[i]!);
    ssRes += (testY[i]! - predicted) ** 2;
    ssTot += (testY[i]! - mean) ** 2;
  }

  return 1 - ssRes / ssTot;
}

function chooseBestModel(candidates: { type: string; score: number; model: { predict(x: number): number } }[]) {
  const best = candidates[0]!;

  // Simplicity bias:
  // if linear is within 0.02 R² of winner, use linear.

  const linear = candidates.find((c) => c.type === 'linear');

  if (linear && best.score - linear.score < 0.02) {
    return linear;
  }

  return best;
}

function describeTrend(data: DataPoint[], title: string): string {
  if (data.length < 5) {
    return `There is not enough data to programatically determine a trend.`;
  }

  // Remove outliers before fitting data to a trend
  const isolationForest = new IsolationForest();
  isolationForest.fit(data); 
  const trainingScores = isolationForest.scores();
  console.log(title, " outlier scores ", trainingScores, " data points ", data)
  const outlierThres = 0.6;
  const outliers = data.filter((_, idx) => trainingScores[idx] >= outlierThres);
  const filteredData = data.filter((_, idx) => trainingScores[idx] < outlierThres);

  const sampled = downsample(filteredData, MAX_SAMPLE_SIZE);
  
  const shuffled = [...sampled].sort(() => Math.random() - 0.5);
  const splitIndex = Math.floor(shuffled.length * 0.8);

  const train = shuffled.slice(0, splitIndex);
  const test = shuffled.slice(splitIndex);

  const trainX = train.map((p) => p.x);
  const trainY = train.map((p) => p.y);

  const testX = test.map((p) => p.x);
  const testY = test.map((p) => p.y);

  const candidates: {
    type: string;
    model: any;
    score: number;
  }[] = [];

  // Linear
  const linModel = new SimpleLinearRegression(trainX, trainY);
  candidates.push({
    type: 'linear',
    model: linModel,
    score: r2(linModel, testX, testY),
  });

  // Polynomial
  try {
    for (let degree = 2; degree <= MAX_POLYNOMIAL_DEGREE; degree += 1) {
      const polyModel = new PolynomialRegression(trainX, trainY, degree);
      const turningPnts = countTurningPnts(polyModel, testX);

      candidates.push({
        type: `${degree}${degree === 2 ? 'nd' : 'th'}-degree-polynomial`,
        model: polyModel,
        score:
          turningPnts === degree - 1
            ? r2(polyModel, testX, testY)
            : Number.MIN_SAFE_INTEGER,
      });
    }
  } catch  {
    // pass, in case PolynomialRegression crashes
  }
  
  // Logarithmic
  const logModel = new SimpleLinearRegression(
    trainX.map((x) => Math.log(x)),
    trainY,
  );
  candidates.push({
    type: 'logarithmic',
    model: logModel,
    score: r2(
      logModel,
      testX.map((x) => Math.log(x)),
      testY,
    ),
  });

  // Exponential
  const expModel = new ExponentialRegression(trainX, trainY);
  candidates.push({
    type: 'exponential',
    model: expModel,
    score: r2(expModel, testX, testY),
  });

  candidates.sort((a, b) => b.score - a.score);

  console.log(title, ' CANDIDATES ', candidates);

  if (!candidates.length) {
    return `Error programmatically fitting the data to a trend.`;
  }

  const best = chooseBestModel(candidates);

  const parts: string[] = [];

  parts.push(`The data best fits a ${best.type} trend.`);

  if (best.score > 0.95) parts.push('Points closely follow the overall trend.');
  else if (best.score > 0.85)
    parts.push('Points generally follow the trend with little variation.');
  else if (best.score > 0.5)
    parts.push('The trend is noticeable but substantial variation exists.');
  else parts.push('The trend is weak and points are widely dispersed.');

  // Confirm whether removed points were outliers
  const residuals = <number[]>[];
  filteredData.forEach(d => {
    const r = Math.abs(d.y - best.model.predict(d.x));
    residuals.push(r);
  })
  console.log(title, " residuals ", residuals)
  const sigma = std(residuals);
  const trueOutliers = outliers.filter(d => {
    const r = Math.abs(d.y - best.model.predict(d.x));
    return r > 3 * sigma;
  })

  if (trueOutliers.length > 0) {
    parts.push(`${trueOutliers.length} outlier point${trueOutliers.length!=1 ? 's were' : ' was'} detected.`)
  }

  return parts.join(' ');
}

export function generateAccessibleChartDescription(
  data: DataPoint[],
  name: string,
  xTitle: string,
  yTitle: string,
): string {
  const parts: string[] = [];

  parts.push(`${name} scatterplot showing ${data.length} data points.`);

  parts.push(
    `The horizontal axis represents ${xTitle} and the vertical axis represents ${yTitle}.`,
  );

  if (data.length > 3) {
    parts.push(describeTrend(data, name));

    const y = data.map((d) => d.y);
    const minY = Math.min(...y);
    const maxY = Math.max(...y);

    const start = y[0]!;
    const end = y[y.length - 1]!;

    parts.push(
      `Y values range from ${minY.toFixed(2)} to ${maxY.toFixed(
        2,
      )}. Initial and final y-values are ${start.toFixed(2)} and ${end.toFixed(2)}, respectively.`,
    );
  }

  return parts.join(' ');
}
