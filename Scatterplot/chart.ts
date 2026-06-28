import Chart from 'chart.js/auto';
import { DataPoint, generateAccessibleChartDescription } from "./chartAltText";
import { anscombeDataset } from './anscombe';

async function createAltText(data: DataPoint[], chartTitle: string) {
    const altText = generateAccessibleChartDescription(data, chartTitle, 'X', 'Y');
    
    const canvasParent = document.getElementById(chartTitle)!;
    const pElem = canvasParent.querySelector('p')!;
    pElem.textContent = altText;
}


function generateChart(data: DataPoint[], chartTitle: string) {
    const canvas = document.createElement('canvas');

    const pElem = document.createElement('p');
    pElem.textContent = "Alt Text Loading."

    const canvasParent = document.createElement('div');
    canvasParent.id = chartTitle;
    canvasParent.append(canvas);
    canvasParent.append(pElem);

    const container = document.getElementById('scatter');
    if (!container) return
    container.append(canvasParent);

    const newChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Data Point',
                    data,
                }
            ]
        }
    });

    createAltText(data, chartTitle);
}
generateChart(anscombeDataset.map(row => ({ x: row.x1, y: row.y1 })), "Dataset 1");
generateChart(anscombeDataset.map(row => ({ x: row.x2, y: row.y2 })), "Dataset 2");
generateChart(anscombeDataset.map(row => ({ x: row.x3, y: row.y3 })), "Dataset 3");
generateChart(anscombeDataset.map(row => ({ x: row.x4, y: row.y4 })), "Dataset 4");
