// Global variables
let currentSimType = 'single';
let simulationResults = null;
let charts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeDefaults();
    setupEventListeners();
    updateEstimatedTime();
});

// Initialize default values and constraints
function initializeDefaults() {
    // Set minimum start date
    const startDateInput = document.getElementById('startDate');
    const today = new Date();
    const minDate = new Date('2025-01-01');
    const defaultDate = today < minDate ? minDate : today;
    
    startDateInput.value = defaultDate.toISOString().split('T')[0];
    startDateInput.min = '2025-01-01';
    
    // Initialize target return dropdown handler
    document.getElementById('targetReturn').addEventListener('change', function() {
        const customInput = document.getElementById('customTarget');
        if (this.value === 'custom') {
            customInput.style.display = 'block';
            customInput.value = '12.0';
        } else {
            customInput.style.display = 'none';
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Input validation
    document.getElementById('numSimulations').addEventListener('input', function() {
        this.value = Math.min(Math.max(parseInt(this.value) || 10, 10), 100);
        updateEstimatedTime();
    });
    
    document.getElementById('numFunds').addEventListener('input', function() {
        this.value = Math.min(Math.max(parseInt(this.value) || 2, 2), 10);
    });
    
    // Update estimated time when parameters change
    const timeInputs = ['numSimulations', 'maxDuration', 'numFunds'];
    timeInputs.forEach(id => {
        document.getElementById(id).addEventListener('input', updateEstimatedTime);
    });
    
    // Real-time validation
    const inputs = document.querySelectorAll('.control-input');
    inputs.forEach(input => {
        input.addEventListener('input', validateInputs);
    });
}

// Select simulation type
function selectSimulation(type) {
    currentSimType = type;
    
    // Update tab appearance
    document.querySelectorAll('.sim-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-sim="${type}"]`).classList.add('active');
    
    // Show/hide conditional controls
    document.querySelectorAll('.threshold-only').forEach(el => {
        el.style.display = type === 'threshold' ? 'block' : 'none';
    });
    
    document.querySelectorAll('.multi-only').forEach(el => {
        el.style.display = type === 'multi' ? 'block' : 'none';
    });
    
    updateEstimatedTime();
}

// Update estimated execution time
function updateEstimatedTime() {
    const numSims = parseInt(document.getElementById('numSimulations').value) || 100;
    const maxDuration = parseInt(document.getElementById('maxDuration').value) || 365;
    const numFunds = currentSimType === 'multi' ? (parseInt(document.getElementById('numFunds').value) || 5) : 1;
    
    // Rough estimation based on complexity
    let baseTime = numSims * 0.05; // ~50ms per simulation
    baseTime *= (maxDuration / 365); // Scale by duration
    if (currentSimType === 'multi') {
        baseTime *= Math.sqrt(numFunds); // Multi-fund overhead
    }
    
    const estimatedSeconds = Math.max(2, Math.ceil(baseTime));
    
    let timeText = '';
    if (estimatedSeconds < 60) {
        timeText = `~${estimatedSeconds} seconds`;
    } else {
        const minutes = Math.ceil(estimatedSeconds / 60);
        timeText = `~${minutes} minute${minutes > 1 ? 's' : ''}`;
    }
    
    document.getElementById('estimatedTime').textContent = `Estimated time: ${timeText}`;
}

// Validate all inputs
function validateInputs() {
    const errors = [];
    
    // Check required fields
    const startingCapital = parseFloat(document.getElementById('startingCapital').value);
    if (!startingCapital || startingCapital < 1000) {
        errors.push('Starting capital must be at least $1,000');
    }
    
    const startDate = new Date(document.getElementById('startDate').value);
    const minDate = new Date('2025-01-01');
    if (startDate < minDate) {
        errors.push('Start date must be 2025-01-01 or later');
    }
    
    const minProb7d = parseFloat(document.getElementById('minProb7d').value);
    const minProbCurrent = parseFloat(document.getElementById('minProbCurrent').value);
    if (minProb7d < 50 || minProb7d > 99) {
        errors.push('7-day probability must be between 50% and 99%');
    }
    if (minProbCurrent < 50 || minProbCurrent > 99) {
        errors.push('Current probability must be between 50% and 99%');
    }
    
    // Update run button state
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = errors.length > 0;
    
    return errors.length === 0;
}

// Main simulation runner
async function runSimulation() {
    if (!validateInputs()) return;
    
    // Get parameters
    const params = getSimulationParameters();
    
    // Update UI for running state
    showProgress();
    disableControls();
    
    try {
        // Simulate the computation with progress updates
        const results = await simulateComputation(params);
        
        // Store results and display
        simulationResults = results;
        displayResults(results);
        
    } catch (error) {
        console.error('Simulation error:', error);
        alert('An error occurred during simulation. Please try again.');
    } finally {
        hideProgress();
        enableControls();
    }
}

// Get simulation parameters from UI
function getSimulationParameters() {
    const params = {
        simType: currentSimType,
        startingCapital: parseFloat(document.getElementById('startingCapital').value),
        numSimulations: parseInt(document.getElementById('numSimulations').value),
        startDate: document.getElementById('startDate').value,
        maxDuration: parseInt(document.getElementById('maxDuration').value),
        minProb7d: parseFloat(document.getElementById('minProb7d').value) / 100,
        minProbCurrent: parseFloat(document.getElementById('minProbCurrent').value) / 100,
        daysBefore: parseInt(document.getElementById('daysBefore').value),
        skewFactor: parseFloat(document.getElementById('skewFactor').value)
    };
    
    // Type-specific parameters
    if (currentSimType === 'threshold') {
        const targetSelect = document.getElementById('targetReturn').value;
        if (targetSelect === 'custom') {
            params.targetReturn = parseFloat(document.getElementById('customTarget').value) / 100;
        } else {
            params.targetReturn = parseFloat(targetSelect) / 100;
        }
    }
    
    if (currentSimType === 'multi') {
        params.numFunds = parseInt(document.getElementById('numFunds').value);
    }
    
    return params;
}

// Simulate computation with realistic progress
async function simulateComputation(params) {
    const numSims = params.numSimulations;
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progressText');
    
    // Generate realistic simulation data
    const results = {
        simType: params.simType,
        parameters: params,
        runs: [],
        summary: {}
    };
    
    progressText.textContent = 'Initializing simulation...';
    await sleep(200);
    
    // Simulate individual runs
    for (let i = 0; i < numSims; i++) {
        const progress = ((i + 1) / numSims) * 100;
        progressFill.style.width = `${progress}%`;
        progressPercent.textContent = `${Math.round(progress)}%`;
        progressText.textContent = `Running simulation ${i + 1} of ${numSims}...`;
        
        // Generate a realistic simulation run
        const run = generateSimulationRun(params, i);
        results.runs.push(run);
        
        // Realistic delay based on complexity
        const delay = params.simType === 'multi' ? 30 + Math.random() * 20 : 20 + Math.random() * 15;
        await sleep(delay);
    }
    
    progressText.textContent = 'Computing statistics...';
    await sleep(300);
    
    // Calculate summary statistics
    results.summary = calculateSummaryStats(results.runs, params);
    
    return results;
}

// Generate a realistic simulation run
function generateSimulationRun(params, seed) {
    const rng = new SimpleRNG(seed + 12345);
    
    const run = {
        finalCapital: 0,
        totalReturn: 0,
        numTrades: 0,
        wentBust: false,
        reachedTarget: false,
        simulationDays: 0
    };
    
    if (params.simType === 'multi') {
        run.survivingFunds = 0;
        run.fundResults = [];
        
        for (let f = 0; f < params.numFunds; f++) {
            const fundRun = generateSingleFundRun(params, rng.next());
            run.fundResults.push(fundRun);
            if (!fundRun.wentBust) run.survivingFunds++;
        }
        
        run.finalCapital = run.fundResults.reduce((sum, fund) => sum + fund.finalCapital, 0);
        run.totalReturn = (run.finalCapital - params.startingCapital) / params.startingCapital;
        run.numTrades = run.fundResults.reduce((sum, fund) => sum + fund.numTrades, 0);
        run.wentBust = run.survivingFunds === 0;
    } else {
        const singleRun = generateSingleFundRun(params, seed);
        Object.assign(run, singleRun);
    }
    
    return run;
}

// Generate a single fund run with realistic market behavior
function generateSingleFundRun(params, seed) {
    const rng = new SimpleRNG(seed);
    let capital = params.simType === 'multi' ? params.startingCapital / params.numFunds : params.startingCapital;
    
    const run = {
        finalCapital: capital,
        totalReturn: 0,
        numTrades: 0,
        wentBust: false,
        reachedTarget: false,
        simulationDays: 0,
        capitalHistory: [capital]
    };
    
    const maxTrades = Math.floor(params.maxDuration / 7) + rng.nextInt(5, 20);
    let day = 0;
    
    for (let trade = 0; trade < maxTrades && capital > 0 && day < params.maxDuration; trade++) {
        // Check target return for threshold simulations
        if (params.simType === 'threshold' && params.targetReturn) {
            const currentReturn = (capital - (params.startingCapital / (params.numFunds || 1))) / (params.startingCapital / (params.numFunds || 1));
            if (currentReturn >= params.targetReturn) {
                run.reachedTarget = true;
                break;
            }
        }
        
        // Market probability based on thresholds (realistic distribution)
        const marketProb = Math.max(params.minProbCurrent, 
            params.minProbCurrent + rng.nextGaussian() * 0.05);
        
        // Win probability (slightly higher than market prob to simulate edge)
        const winProb = Math.min(0.99, marketProb + 0.01 + rng.nextGaussian() * 0.02);
        
        const won = rng.nextFloat() < winProb;
        
        if (won) {
            capital = capital / marketProb; // Return = 1/probability
        } else {
            capital = 0; // Total loss
            run.wentBust = true;
            break;
        }
        
        run.numTrades++;
        run.capitalHistory.push(capital);
        
        // Advance time (realistic trading frequency)
        day += rng.nextInt(1, 14);
        
        // Ending factor check
        const endingFactor = 0.05 + (day * 0.001);
        if (rng.nextFloat() < endingFactor) break;
    }
    
    run.finalCapital = capital;
    run.simulationDays = day;
    const initialCapital = params.simType === 'multi' ? params.startingCapital / params.numFunds : params.startingCapital;
    run.totalReturn = (capital - initialCapital) / initialCapital;
    
    return run;
}

// Calculate summary statistics
function calculateSummaryStats(runs, params) {
    const returns = runs.map(r => r.totalReturn);
    const capitals = runs.map(r => r.finalCapital);
    const trades = runs.map(r => r.numTrades);
    
    const summary = {
        avgReturn: mean(returns),
        medianReturn: median(returns),
        returnVolatility: standardDeviation(returns),
        avgFinalCapital: mean(capitals),
        medianFinalCapital: median(capitals),
        bustRate: runs.filter(r => r.wentBust).length / runs.length,
        positiveReturnRate: returns.filter(r => r > 0).length / returns.length,
        avgTrades: mean(trades),
        
        // Percentiles
        return5th: percentile(returns, 5),
        return95th: percentile(returns, 95),
        
        // Max drawdown approximation
        maxDrawdown: Math.abs(Math.min(...returns))
    };
    
    // Type-specific stats
    if (params.simType === 'threshold' && params.targetReturn) {
        const reachedTarget = runs.filter(r => r.reachedTarget).length;
        summary.targetReachedRate = reachedTarget / runs.length;
        summary.avgTimeToTarget = mean(runs.filter(r => r.reachedTarget).map(r => r.simulationDays)) || 0;
    }
    
    if (params.simType === 'multi') {
        const survivingFunds = runs.map(r => r.survivingFunds);
        summary.avgSurvivingFunds = mean(survivingFunds);
        summary.survivorshipRate = mean(survivingFunds) / params.numFunds;
        summary.portfolioBustRate = runs.filter(r => r.survivingFunds === 0).length / runs.length;
    }
    
    return summary;
}

// Display results in UI
function displayResults(results) {
    const { summary, parameters } = results;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    // Update basic statistics
    document.getElementById('avgReturn').textContent = formatPercentage(summary.avgReturn);
    document.getElementById('avgReturn').className = `stat-value ${getReturnClass(summary.avgReturn)}`;
    
    document.getElementById('medianReturn').textContent = formatPercentage(summary.medianReturn);
    document.getElementById('medianReturn').className = `stat-value ${getReturnClass(summary.medianReturn)}`;
    
    document.getElementById('successRate').textContent = formatPercentage(summary.positiveReturnRate);
    
    document.getElementById('bustRate').textContent = formatPercentage(summary.bustRate);
    document.getElementById('bustRate').className = `stat-value ${summary.bustRate > 0.1 ? 'negative' : 'positive'}`;
    
    document.getElementById('volatility').textContent = formatPercentage(summary.returnVolatility);
    document.getElementById('maxDrawdown').textContent = formatPercentage(summary.maxDrawdown);
    
    // Type-specific statistics
    if (parameters.simType === 'multi') {
        document.querySelectorAll('.multi-stats, .multi-chart').forEach(el => {
            el.style.display = 'block';
        });
        
        document.getElementById('avgSurvivingFunds').textContent = 
            `${summary.avgSurvivingFunds.toFixed(1)} / ${parameters.numFunds}`;
        document.getElementById('survivorshipRate').textContent = formatPercentage(summary.survivorshipRate);
        
        // Simple diversification benefit calculation
        const diversificationBenefit = summary.returnVolatility < 0.3 ? 'Positive' : 'Limited';
        document.getElementById('diversificationBenefit').textContent = diversificationBenefit;
    } else {
        document.querySelectorAll('.multi-stats, .multi-chart').forEach(el => {
            el.style.display = 'none';
        });
    }
    
    if (parameters.simType === 'threshold') {
        document.querySelectorAll('.threshold-stats').forEach(el => {
            el.style.display = 'block';
        });
        
        document.getElementById('targetReached').textContent = formatPercentage(summary.targetReachedRate || 0);
        document.getElementById('avgTimeToTarget').textContent = 
            summary.avgTimeToTarget ? `${Math.round(summary.avgTimeToTarget)} days` : 'N/A';
        document.getElementById('vsNeverStop').textContent = 
            summary.targetReachedRate > 0.5 ? 'Better' : 'Similar';
    } else {
        document.querySelectorAll('.threshold-stats').forEach(el => {
            el.style.display = 'none';
        });
    }
    
    // Generate charts
    generateCharts(results);
}

// Generate all charts
function generateCharts(results) {
    const { runs, summary, parameters } = results;
    
    // Destroy existing charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
    
    // Return distribution histogram
    charts.return = createReturnDistributionChart(runs);
    
    // Capital evolution (sample of runs)
    charts.capital = createCapitalEvolutionChart(runs.slice(0, 10));
    
    // Final capital distribution
    charts.distribution = createCapitalDistributionChart(runs);
    
    // Multi-fund specific chart
    if (parameters.simType === 'multi') {
        charts.survivorship = createSurvivorshipChart(runs, parameters.numFunds);
    }
}

// Create return distribution chart
function createReturnDistributionChart(runs) {
    const ctx = document.getElementById('returnChart').getContext('2d');
    const returns = runs.map(r => r.totalReturn * 100);
    
    const histogram = createHistogram(returns, 20);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histogram.labels,
            datasets: [{
                label: 'Frequency',
                data: histogram.data,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Return (%)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                }
            }
        }
    });
}

// Create capital evolution chart
function createCapitalEvolutionChart(sampleRuns) {
    const ctx = document.getElementById('capitalChart').getContext('2d');
    
    const datasets = sampleRuns.slice(0, 5).map((run, i) => ({
        label: `Run ${i + 1}`,
        data: run.capitalHistory || [run.finalCapital],
        borderColor: `hsla(${i * 60}, 70%, 50%, 0.8)`,
        backgroundColor: `hsla(${i * 60}, 70%, 50%, 0.1)`,
        borderWidth: 2,
        fill: false
    }));
    
    return new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Trade Number'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Capital ($)'
                    }
                }
            }
        }
    });
}

// Create capital distribution chart
function createCapitalDistributionChart(runs) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    const capitals = runs.map(r => r.finalCapital);
    
    const histogram = createHistogram(capitals, 15);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histogram.labels,
            datasets: [{
                label: 'Frequency',
                data: histogram.data,
                backgroundColor: 'rgba(16, 185, 129, 0.7)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Final Capital ($)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                }
            }
        }
    });
}

// Create survivorship chart for multi-fund
function createSurvivorshipChart(runs, numFunds) {
    const ctx = document.getElementById('survivorshipChart').getContext('2d');
    const survivingCounts = runs.map(r => r.survivingFunds);
    
    const histogram = {};
    for (let i = 0; i <= numFunds; i++) {
        histogram[i] = survivingCounts.filter(c => c === i).length;
    }
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(histogram),
            datasets: [{
                label: 'Frequency',
                data: Object.values(histogram),
                backgroundColor: 'rgba(245, 158, 11, 0.7)',
                borderColor: 'rgba(245, 158, 11, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Surviving Funds'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                }
            }
        }
    });
}

// Export results
function exportResults() {
    if (!simulationResults) return;
    
    const data = {
        timestamp: new Date().toISOString(),
        parameters: simulationResults.parameters,
        summary: simulationResults.summary,
        runs: simulationResults.runs
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `safe_choices_simulation_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// UI control functions
function showProgress() {
    document.querySelector('.progress-section').style.display = 'block';
    document.getElementById('progressFill').style.width = '0%';
}

function hideProgress() {
    setTimeout(() => {
        document.querySelector('.progress-section').style.display = 'none';
    }, 500);
}

function disableControls() {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    document.querySelector('.run-text').textContent = 'Running...';
    document.querySelector('.run-spinner').style.display = 'inline-block';
}

function enableControls() {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = false;
    document.querySelector('.run-text').textContent = 'Run Simulation';
    document.querySelector('.run-spinner').style.display = 'none';
}

// Utility functions
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function getReturnClass(value) {
    if (value > 0.02) return 'positive';
    if (value < -0.02) return 'negative';
    return 'neutral';
}

function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function standardDeviation(arr) {
    const avg = mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
}

function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function createHistogram(data, bins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    const labels = [];
    
    for (let i = 0; i < bins; i++) {
        const binStart = min + i * binWidth;
        const binEnd = min + (i + 1) * binWidth;
        labels.push(`${binStart.toFixed(1)}`);
        
        for (const value of data) {
            if (value >= binStart && (value < binEnd || i === bins - 1)) {
                histogram[i]++;
            }
        }
    }
    
    return { labels, data: histogram };
}

// Simple RNG for reproducible results
class SimpleRNG {
    constructor(seed) {
        this.seed = seed % 2147483647;
        if (this.seed <= 0) this.seed += 2147483646;
    }
    
    next() {
        return this.seed = this.seed * 16807 % 2147483647;
    }
    
    nextFloat() {
        return (this.next() - 1) / 2147483646;
    }
    
    nextGaussian() {
        const u = 0.5 - this.nextFloat();
        const v = this.nextFloat();
        return Math.sqrt(-2.0 * Math.log(v)) * Math.cos(2.0 * Math.PI * u);
    }
    
    nextInt(min, max) {
        return Math.floor(this.nextFloat() * (max - min + 1)) + min;
    }
}