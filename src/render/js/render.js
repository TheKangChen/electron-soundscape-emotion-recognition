const { Menu, dialog, app } = require('@electron/remote');
const { ipcRenderer } = require('electron');

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const parse = require('csv-parse/sync');


const debug = true;

let featureContainer; // container for all feature
let arousal; // arousal score
let valence; // valence score

let file; // csv file
let featureData; // csv feature data



/**********************************************/
// Select HTML elements
/**********************************************/
const fileBtn = document.getElementById("fileBtn");
fileBtn.onclick = getCSVfile;

const predictBtn = document.getElementById('predictBtn');
predictBtn.onclick = predict;

const saveBtn = document.getElementById("saveBtn");
saveBtn.onclick = saveAsCSV;



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
async function getCSVfile() {
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

        file = selectedFile.filePaths[0].toString();
        const canceled = selectedFile.canceled;
        // const filePath = file[0].toString();

        if (debug) {
            console.log(file);
            console.log(canceled);
        }

        if (file) {
            fs.readFile(file, 'utf8', (err, data) => {
                console.log(data.split(','));
                featureData = data.split(',').map(x => parseFloat(x));
                debug ? console.log(featureData) : '';
            });
        }
    } catch (err) {
        console.log(err);
  }
}


// save feature of audio file to csv
async function saveAsCSV() {
    const featureStats = getStats(featureContainer);
    debug ? console.log(featureStats.length) : '';

    if (featureStats) {
        const data = featureStats.toString();
        // save file
        const filename = file[0].split('/').slice(-1).toString();
        debug ? console.log(filename, typeof filename) : '';

        try {
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
}



/****************** Model Prediction *******************/
// predict
function predict() {
    // aroModel.;
    // valModel;
}
