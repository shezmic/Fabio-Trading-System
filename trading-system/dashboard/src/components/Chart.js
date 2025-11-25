import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

const Chart = ({ data }) => {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const seriesRef = useRef();

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            width: 800,
            height: 400,
            layout: {
                backgroundColor: '#000000',
                textColor: '#d1d4dc',
            },
            grid: {
                vertLines: { color: '#404040' },
                horzLines: { color: '#404040' },
            },
        });

        const candleSeries = chart.addCandlestickSeries();
        seriesRef.current = candleSeries;
        chartRef.current = chart;

        return () => {
            chart.remove();
        };
    }, []);

    useEffect(() => {
        if (seriesRef.current && data) {
            // seriesRef.current.setData(data);
        }
    }, [data]);

    return <div ref={chartContainerRef} />;
};

export default Chart;
