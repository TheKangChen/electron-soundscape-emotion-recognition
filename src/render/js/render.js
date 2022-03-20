const { Menu, dialog, app } = require('@electron/remote');
const { ipcRenderer } = require('electron');

const fs = require('fs');
const path = require('path');
const wav = require('node-wav');
const Meyda = require('meyda');
const tf = require('@tensorflow/tfjs-node');


const debug = true;


/**************** Variables *******************/
// List of features
let featuresList = [
    'rms',
    'zcr',
    'spectralRolloff',
    'spectralCentroid',
    'spectralSpread',
    'spectralSkewness',
    'spectralKurtosis',
    'spectralFlatness',
    'mfcc',
    'chroma',
    'loudness',
    'energy',
    'perceptualSharpness',
    'spectralSlope'
];
const finalFeatureSet = [
    'rms_mean',
    'rms_std',
    'zerocross_mean',
    'zerocross_std',
    'rolloff_mean',
    'rolloff_std',
    'centroid_mean',
    'centroid_std',
    'spread_mean',
    'spread_std',
    'skewness_mean',
    'skewness_std',
    'kurtosis_mean',
    'kurtosis_std',
    'flatness_mean',
    'flatness_std',
    'mfcc_mean_1',
    'mfcc_mean_2',
    'mfcc_mean_3',
    'mfcc_mean_4',
    'mfcc_mean_5',
    'mfcc_mean_6',
    'mfcc_mean_7',
    'mfcc_mean_8',
    'mfcc_mean_9',
    'mfcc_mean_10',
    'mfcc_mean_11',
    'mfcc_mean_12',
    'mfcc_mean_13',
    'mfcc_std_1',
    'mfcc_std_2',
    'mfcc_std_3',
    'mfcc_std_4',
    'mfcc_std_5',
    'mfcc_std_6',
    'mfcc_std_7',
    'mfcc_std_8',
    'mfcc_std_9',
    'mfcc_std_10',
    'mfcc_std_11',
    'mfcc_std_12',
    'mfcc_std_13',
    'chromagram_mean_1',
    'chromagram_mean_2',
    'chromagram_mean_3',
    'chromagram_mean_4',
    'chromagram_mean_5',
    'chromagram_mean_6',
    'chromagram_mean_7',
    'chromagram_mean_8',
    'chromagram_mean_9',
    'chromagram_mean_10',
    'chromagram_mean_11',
    'chromagram_mean_12',
    'chromagram_std_1',
    'chromagram_std_2',
    'chromagram_std_3',
    'chromagram_std_4',
    'chromagram_std_5',
    'chromagram_std_6',
    'chromagram_std_7',
    'chromagram_std_8',
    'chromagram_std_9',
    'chromagram_std_10',
    'chromagram_std_11',
    'chromagram_std_12',
    'loudness_mean',
    'loudness_std',
    'energy_mean',
    'energy_std',
    'perceptual_sharp_mean',
    'perceptual_sharp_std',
    'spectral_slope_mean',
    'spectral_slope_std'
];

// file io
let audioFile; // audio file path
let csvFile; // csv file
let featureData; // csv read feature data

let fileDir; // directory containing all the audio files

// audio stream
let fileBuffer; // audio file buffer
let audioData; // audio file data
let signal; // audio signal from audio data

// feature extraction
const INPUT_DIM = 74;
// let featureContainer = []; // container for all feature

// model Prediction
let arousal; // arousal score
let valence; // valence score

// audio metadata
const sampleRate = 44100;
const bufferSize = 1024; // 23ms
const hopSize = bufferSize / 2;
const windowingFunction = 'hanning';
const n_mfcc = 13;


/**********************************************/
// Select HTML elements
/**********************************************/
const audioElement = document.getElementById('audio');

const fileBtn = document.getElementById("fileBtn");
// fileBtn.onclick = getAudioFile;
fileBtn.onclick = getFileDir;

const csvBtn = document.getElementById("csvBtn");
csvBtn.onclick = getCSVFile;

const extractBtn = document.getElementById("extractBtn");
extractBtn.onclick = getFeatures;

const predictBtn = document.getElementById('predictBtn');
predictBtn.onclick = predict;

const saveFeatBtn = document.getElementById("saveFeatBtn");
saveFeatBtn.onclick = saveFeatureToCSV;

const savePredBtn = document.getElementById("savePredBtn");
savePredBtn.onclick = savePredictionToCSV;



/**********************************************/
// Load Models
/**********************************************/
const aroModel = await tf.loadLayersModel('../models/arousal/model.json');
const valModel = await tf.loadLayersModel('../models/valence/model.json');



/**********************************************/
// Fuctions
/**********************************************/

/****************** File io *******************/
// get audio file
async function getAudioFile() {
    try {
        const selectedFile = await dialog.showOpenDialog({
            title: 'Open File',
            defaultPath: app.getPath('home') || path.parse(process.cwd()).root,
            buttonLabel: 'Open',
            properties: ['openFile', 'openDirectory'],
            filters: [
                {
                    name: 'Audios',
                    extensions: ['wav']
                }
            ]
        });
        audioFile = selectedFile.filePaths;
        const canceled = selectedFile.canceled;

        if (debug) {
            console.log(audioFile);
            console.log(canceled);
        }
    } catch (err) {
        console.log(err);
  }
}


// get all files inside directory
async function getFileDir() {
    try {
        const selectedDir = await dialog.showOpenDialog({
            title: 'Open Directory',
            defaultPath: app.getPath('desktop'),
            buttonLabel: 'Open',
            properties: ['openDirectory'],
        });
        fileDir = selectedDir.filePaths;
        const canceled = selectedDir.canceled;
        
        fileBuffer = fs.readdirSync(fileDir[0]);
        if (debug) {
            console.log(fileDir);
            console.log(canceled);
            console.log(fileBuffer);
        }
    } catch (err) {
        console.log(err);
    }
}

// save feature of audio file to csv
async function saveFeatureToCSV() {
    try {
        const featureStats = featureContainer.length != 0 ? getStats(featureContainer) : console.log('No features extracted yet');
        debug ? console.log(featureStats.length) : '';
        const data = featureStats.toString();
        // save file
        const filename = audioFile[0].split('/').slice(-1).toString().slice(0,-4) + '_features';
        debug ? console.log(filename, typeof filename) : '';
        const selectedFolder = await dialog.showSaveDialog({
            title: 'Save File',
            defaultPath: path.join(app.getPath('desktop'), filename) || path.join(__dirname, '../../../export/', filename),
            buttonLabel: 'Save',
            filters: [
                {
                    name: 'CSV Files',
                    extensions: ['csv']
                }
            ],
            properties: [
                'createDirectory',
                'showOverwriteConfirmation'
            ]
        });

        const filePath = selectedFolder.filePath;
        const canceled = selectedFolder.canceled;

        if (debug) {
            console.log(filePath);
            console.log(canceled);
        }

        if (!canceled) {
            // write to csv file
            fs.writeFile(filePath.toString(), data, err => {
                if (err) throw err;
                console.log(err);
            })
        }
    } catch (err) {
        console.log(err);
    }
}


// get feature csv file
async function getCSVFile() {
    try {
        const selectedFile = await dialog.showOpenDialog({
            title: 'Open File',
            defaultPath: app.getPath('home') || path.parse(process.cwd()).root,
            buttonLabel: 'Open',
            properties: ['openFile', 'openDirectory'],
            filters: [
                {
                    name: 'CSV Files',
                    extensions: ['csv']
                }
            ]
        });

        csvFile = selectedFile.filePaths[0].toString();
        const canceled = selectedFile.canceled;
        // const filePath = file[0].toString();

        if (debug) {
            console.log(csvFile);
            console.log(canceled);
        }

        if (csvFile) {
            fs.readFile(csvFile, 'utf8', (err, data) => {
                console.log(data.split(','));
                featureData = data.split(',').map(x => parseFloat(x));
                debug ? console.log(featureData) : '';
            });
        }
    } catch (err) {
        console.log(err);
  }
}


// save prediction of audio file to csv
async function savePredictionToCSV() {
    const prediction = (arousal && valence) ? [arousal[0], valence[0]] : console.log('No predictions yet');
    debug ? console.log(prediction) : '';
    try {
        const data = prediction.toString();
        debug ? console.log(data, typeof data) : '';
        // get filename without path and extension
        const filename = csvFile.split('/').slice(-1).toString().slice(0,-4) + '_prediction';
        debug ? console.log(filename, typeof filename) : '';
        const selectedFolder = await dialog.showSaveDialog({
            title: 'Save File',
            defaultPath: path.join(app.getPath('desktop'), filename) || path.join(__dirname, '../../../export/', filename),
            buttonLabel: 'Save',
            filters: [
                {
                    name: 'CSV Files',
                    extensions: ['csv']
                }
            ],
            properties: [
                'createDirectory',
                'showOverwriteConfirmation'
            ]
        });

        const filePath = selectedFolder.filePath;
        const canceled = selectedFolder.canceled;

        if (debug) {
            console.log(filePath);
            console.log(canceled);
        }

        if (!canceled) {
            // write to csv file
            fs.writeFile(filePath.toString(), data, err => {
                if (err) throw err;
                console.log(err);
            })
        }
    } catch (err) {
        console.log(err);
    }
}



/****************** Features ******************/
// get features
async function getFeatures() {
    // config Meyda
    Meyda.sampleRate = sampleRate;
    Meyda.windowingFunction = windowingFunction;
    Meyda.numberOfMFCCCoefficients = n_mfcc;

    try {
        let allfilesFeatureStats = [];

        fileBuffer.forEach(e => {
            // read wav file
            const path = fileDir + '/' + e;
            const buffer = fs.readFileSync(path);
            audioData = wav.decode(buffer);
            signal = audioData.channelData[0];

            // check if signal length is the power of 2
            let paddedSig;
            if (!isPowerOf2(signal.length)) {
                const len = signal.length;
                const targetPower = Math.ceil(Math.log2(len));
                const newLen = Math.pow(2, targetPower);
                const truncLen = Math.pow(2, (targetPower - 1));

                if ((newLen - len) < (len - truncLen)) {
                    const padLen = newLen - len;
                    const zeros = new Float32Array(padLen);

                    paddedSig = new Float32Array(newLen);
                    paddedSig.set(signal);
                    paddedSig.set(zeros, len);
                } else {
                    paddedSig = signal.subarray(0, truncLen);
                }
            } else {
                paddedSig = signal;
            }
            // extract through signal
            let featureContainer = [];

            const sigLen = paddedSig.length;
            for (let i=0; i<sigLen; i+=bufferSize) {
                const currentSig = paddedSig.subarray(i, i+bufferSize)
                let extractedFeatures = Meyda.extract(featuresList, currentSig);
                featureContainer.push(extractedFeatures);

                // debug ? console.log(extractedFeatures) : '';
            }
            
            if (debug) {
                console.log(e);
                // console.log(featureContainer);
                console.log(featureContainer.length);
            }

            const featureStats = featureContainer.length != 0 ? getStats(featureContainer) : console.log('No features extracted yet');
            allfilesFeatureStats.push(featureStats);
        })
        const normalizedAllFilesFeature = normalizeFeature(allfilesFeatureStats);
        
        if (debug) {
            console.log('all feature stats', allfilesFeatureStats);
            console.log('normalized all feature stats', normalizedAllFilesFeature);
        }

    } catch (err) {
        console.log(err);
    }
}


// get Mean & Std of features
function getStats(featureContainer) {
    /* 
    featureContainer: [
        {rms, zcr, spectralRolloff, ...}
        {rms, zcr, spectralRolloff, ...}
        {rms, zcr, spectralRolloff, ...}
        .
        .
        .
    ] // time series of extracted feature objects

    *****************************
    return: Float32Array(n_features * 2)
        [ rms_mean, rms_std, zcr_mean, zcr_std, spectralRolloff_mean, spectralRolloff_std, ... ]
    */

    if (!Array.isArray(featureContainer)) {
        throw 'Cannot get stats, getStats() parameter 0 not an array';
    }
    const len = featureContainer.length;
    const n = finalFeatureSet.length;
    // debug ? console.log(len, n) : '';

    // Put features into their corresponding array
    let stats = [];
    let featureSet = {
        rms: [],
        zcr: [],
        rolloff: [],
        centroid: [],
        spread: [],
        skewness: [],
        kurtosis: [],
        flatness: [],
        mfcc1: [],
        mfcc2: [],
        mfcc3: [],
        mfcc4: [],
        mfcc5: [],
        mfcc6: [],
        mfcc7: [],
        mfcc8: [],
        mfcc9: [],
        mfcc10: [],
        mfcc11: [],
        mfcc12: [],
        mfcc13: [],
        chroma1: [],
        chroma2: [],
        chroma3: [],
        chroma4: [],
        chroma5: [],
        chroma6: [],
        chroma7: [],
        chroma8: [],
        chroma9: [],
        chroma10: [],
        chroma11: [],
        chroma12: [],
        loudness: [],
        energy: [],
        sharpness: [],
        spectSlope: [],
    } // 37

    featureContainer.forEach(e => {
        featureSet.rms.push(e.rms);
        featureSet.zcr.push(e.zcr);
        featureSet.rolloff.push(e.spectralRolloff);
        featureSet.centroid.push(e.spectralCentroid);
        featureSet.spread.push(e.spectralSpread);
        featureSet.skewness.push(e.spectralSkewness);
        featureSet.kurtosis.push(e.spectralKurtosis);
        featureSet.flatness.push(e.spectralFlatness);
        featureSet.mfcc1.push(e.mfcc[0]);
        featureSet.mfcc2.push(e.mfcc[1]);
        featureSet.mfcc3.push(e.mfcc[2]);
        featureSet.mfcc4.push(e.mfcc[3]);
        featureSet.mfcc5.push(e.mfcc[4]);
        featureSet.mfcc6.push(e.mfcc[5]);
        featureSet.mfcc7.push(e.mfcc[6]);
        featureSet.mfcc8.push(e.mfcc[7]);
        featureSet.mfcc9.push(e.mfcc[8]);
        featureSet.mfcc10.push(e.mfcc[9]);
        featureSet.mfcc11.push(e.mfcc[10]);
        featureSet.mfcc12.push(e.mfcc[11]);
        featureSet.mfcc13.push(e.mfcc[12]);
        featureSet.chroma1.push(e.chroma[0]);
        featureSet.chroma2.push(e.chroma[1]);
        featureSet.chroma3.push(e.chroma[2]);
        featureSet.chroma4.push(e.chroma[3]);
        featureSet.chroma5.push(e.chroma[4]);
        featureSet.chroma6.push(e.chroma[5]);
        featureSet.chroma7.push(e.chroma[6]);
        featureSet.chroma8.push(e.chroma[7]);
        featureSet.chroma9.push(e.chroma[8]);
        featureSet.chroma10.push(e.chroma[9]);
        featureSet.chroma11.push(e.chroma[10]);
        featureSet.chroma12.push(e.chroma[11]);
        featureSet.loudness.push(e.loudness.total / 24);
        featureSet.energy.push(e.energy);
        featureSet.sharpness.push(e.perceptualSharpness);
        featureSet.spectSlope.push(e.spectralSlope);
    })
    // debug ? console.log(featureSet) : '';
    
    // Get mean and std of each feature
    for (let i=0; i<n/2; ++i) {
        stats.push(mean(featureSet[Object.keys(featureSet)[i]]));
        stats.push(std(featureSet[Object.keys(featureSet)[i]]));
    }
    // debug ? console.log(stats) : '';
    
    // Return array of feature statistics as Float32Array
    return new Float32Array(stats)
}


// normalize feature set
function normalizeFeature(allfeatureStats) {
    let max = new Array(INPUT_DIM).fill(0);
    let min = new Array(INPUT_DIM).fill(0);

    // get max of indices 0 - 73 of all array
    const len = allfeatureStats.length;
    for (let i=0; i<len; ++i) {
        for (let j=0; j<INPUT_DIM; ++j) {
            if (allfeatureStats[i][j] > max[j]) max[j] = allfeatureStats[i][j];
            if (allfeatureStats[i][j] < min[j]) min[j] = allfeatureStats[i][j];
        }
    }
    // normalize data base on the max of each index
    const normalized = allfeatureStats.map(array => {
        return array.map((n, i) => {
            const norm = max[i] - min[i];
            return (n + min[i]) / norm;
        });
    });

    if (debug) {
        console.log(max);
        console.log(min);
    }

    return normalized
}



/************** Model Prediction **************/
// predict
async function predict() {
    try {
        const input = featureData ? tf.tensor(featureData).reshape([1,74]) : console.log('No feautre data selected');
        arousal = await aroModel.predict(input).data();
        valence = await valModel.predict(input).data();
        debug ? console.log(arousal, valence) : '';
    } catch (err) {
        console.log(err);
    }
}



/************* Math Calculations **************/
// check if number is the power of 2
function isPowerOf2(v) {
    return v && !(v & (v - 1));
}


// normalize array of feature by the largest value
function normalize(a) {
    const max = Math.max(a);
    return a.map(x => x / max);
}


// mean of an array
function mean(a) {
    if (!Array.isArray(a)) throw 'mean() parameter 0 not an array';
    let n = a.length;
    if (n === 0) return 0;
    return (a.reduce((prev, curr) => prev + curr) / n);
}


// standard deviation of an array
function std(a) {
    if (!Array.isArray(a)) throw 'std() parameter 0 not an array';
    const n = a.length;
    if (n === 0) return 0;
    const m = a.reduce((prev, curr) => prev + curr) / n; // calculate mean
    return Math.sqrt(a.map(x => Math.pow(x - m, 2)).reduce((prev, curr) => prev + curr) / n);
}
