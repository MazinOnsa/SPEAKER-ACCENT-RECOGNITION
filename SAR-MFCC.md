                                       Speaker Accent Recognition Using MFCCs
                                      -----------------------------------------

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Domain-Knowledge" data-toc-modified-id="Domain-Knowledge-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Domain Knowledge</a></span></li><li><span><a href="#Dataset" data-toc-modified-id="Dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Dataset</a></span><ul class="toc-item"><li><span><a href="#Source" data-toc-modified-id="Source-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Source</a></span></li><li><span><a href="#Problem-formulation" data-toc-modified-id="Problem-formulation-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Problem formulation</a></span></li></ul></li><li><span><a href="#Related-Work" data-toc-modified-id="Related-Work-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Related Work</a></span></li><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Exploration</a></span></li><li><span><a href="#Data-Distribution" data-toc-modified-id="Data-Distribution-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data Distribution</a></span></li><li><span><a href="#Outliers-Detection" data-toc-modified-id="Outliers-Detection-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Outliers Detection</a></span><ul class="toc-item"><li><span><a href="#Interquartile-Range-(IQR)" data-toc-modified-id="Interquartile-Range-(IQR)-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Interquartile Range (IQR)</a></span></li><li><span><a href="#Mahalanabois-Distance" data-toc-modified-id="Mahalanabois-Distance-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Mahalanabois Distance</a></span></li><li><span><a href="#Outlier-Removal-Tradeoff" data-toc-modified-id="Outlier-Removal-Tradeoff-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Outlier Removal Tradeoff</a></span></li></ul></li><li><span><a href="#Multicollinearity-Detection" data-toc-modified-id="Multicollinearity-Detection-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Multicollinearity Detection</a></span><ul class="toc-item"><li><span><a href="#Pearson-correlation" data-toc-modified-id="Pearson-correlation-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Pearson correlation</a></span></li><li><span><a href="#Hieracichal-clustering" data-toc-modified-id="Hieracichal-clustering-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Hieracichal clustering</a></span></li></ul></li><li><span><a href="#Dimensionality-Reduction" data-toc-modified-id="Dimensionality-Reduction-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Dimensionality Reduction</a></span></li><li><span><a href="#Classification" data-toc-modified-id="Classification-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Classification</a></span><ul class="toc-item"><li><span><a href="#Imbalanced-Dataset-Classification" data-toc-modified-id="Imbalanced-Dataset-Classification-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Imbalanced Dataset Classification</a></span></li><li><span><a href="#Classification-Metric" data-toc-modified-id="Classification-Metric-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Classification Metric</a></span></li><li><span><a href="#K-fold-Cross-Validation" data-toc-modified-id="K-fold-Cross-Validation-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>K-fold Cross-Validation</a></span></li><li><span><a href="#Classification-Models" data-toc-modified-id="Classification-Models-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>Classification Models</a></span><ul class="toc-item"><li><span><a href="#K-Nearest-Neighbour:" data-toc-modified-id="K-Nearest-Neighbour:-9.4.1"><span class="toc-item-num">9.4.1&nbsp;&nbsp;</span>K-Nearest Neighbour:</a></span></li><li><span><a href="#Support-Vector-Machine:" data-toc-modified-id="Support-Vector-Machine:-9.4.2"><span class="toc-item-num">9.4.2&nbsp;&nbsp;</span>Support Vector Machine:</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-9.4.3"><span class="toc-item-num">9.4.3&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#Implementation-Framework" data-toc-modified-id="Implementation-Framework-9.4.4"><span class="toc-item-num">9.4.4&nbsp;&nbsp;</span>Implementation Framework</a></span></li><li><span><a href="#Imbalanced-dataset-classification" data-toc-modified-id="Imbalanced-dataset-classification-9.4.5"><span class="toc-item-num">9.4.5&nbsp;&nbsp;</span>Imbalanced dataset classification</a></span></li><li><span><a href="#Imbalanced-dataset-classification-by-modifying-classifiers'-cost-function" data-toc-modified-id="Imbalanced-dataset-classification-by-modifying-classifiers'-cost-function-9.4.6"><span class="toc-item-num">9.4.6&nbsp;&nbsp;</span>Imbalanced dataset classification by modifying classifiers' cost function</a></span></li></ul></li></ul></li><li><span><a href="#Dataset-Balancing:" data-toc-modified-id="Dataset-Balancing:-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Dataset Balancing:</a></span><ul class="toc-item"><li><span><a href="#Synthetic-Minority-Oversampling-Technique-(SMOTE)" data-toc-modified-id="Synthetic-Minority-Oversampling-Technique-(SMOTE)-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Synthetic Minority Oversampling Technique (SMOTE)</a></span></li><li><span><a href="#Adaptive-Synthetic-(ADASYN)" data-toc-modified-id="Adaptive-Synthetic-(ADASYN)-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Adaptive Synthetic (ADASYN)</a></span></li><li><span><a href="#Oversampled-Dataset-Classification" data-toc-modified-id="Oversampled-Dataset-Classification-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Oversampled Dataset Classification</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# Domain Knowledge

Accent recognition is classification of the speaker accent from an input signal. Classifying accents can provide information about a speaker’s nationality and heritage, which can help identify topics more relevant to the user, for the purposes of search results and advertisements. Typical applications include online banking, telephone shopping, and security applications.
Typically, the input signal is represented in frequency domain then dimensionality reduction can be performed together with feature extraction.

<img src="images/1.jpg" />

A common feature extraction technique for the purpose of speech recognition is Mel-frequency cepstral coefficients or MFCCs. MFCCs are coefficients that collectively make up an MFC (mel-frequency cepstrum) which is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
The main idea of MFCC is to transform the signal from time domain to frequency domain and to map the transformed signal in hertz onto Mel-scale due to the fact that 1 kHz is a threshold of humans’ hearing ability.

MFCCs are commonly derived as follows: 

1.	Take the absolute value of the short time Fourier transform of (a windowed excerpt of) a signal.
2.	Map the powers of the spectrum obtained above onto the mel scale
3.	Take the logs of the powers at each of the mel frequencies.
4.	Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
5.	The MFCCs are the amplitudes of the resulting spectrum.
6.	Return the first $q$ MFCCs.

MFCC values are not very robust in the presence of additive noise, and so it is common to normalise their values in speech recognition systems to lessen the influence of noise. (energy terms)


<img src="images/2.png" />


# Dataset
   ##  Source
   
A total of 329 signal data were collected from the voice of 22 speakers 11 female and 11 male of accented speakers speaking English, containing 165 US voice and 164 non-US voice from 5 countries: Spain, France, Germany, Italy, and UK.

The sound tracks have lengths of around 1 second, with a sampling rate of 44,100 Hz, each sound track vector on the time domain has more than 30,000 entries. Because of the method used in collecting the data, there is no background noise in any sound tracks.

The 12th lowest order melfrequency cepstral coefficients (MFCCs) of the audio signals are used as inputs to the algorithms.

The Source of both audio files and MFCC spreadsheet available at UCI [here](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition)


   ## Problem formulation
   
This accent reognition is a classification problem and the response variable $yi$ is givn by:

$$
y_i = \left\{
    \begin{array}\\
        0, \space \space ES\\
        1, \space \space FR\\
        2, \space \space GE\\
        3, \space \space IT\\
        4, \space \space UK\\
        5, \space \space US 
    \end{array}
\right.
$$

showing that there are 6 class labels 

The design is **balanced in terms of US/NOT US accent but we want to extend the problem to classify all accents, hence imbalanced problem**


# Related Work 

[1](https://arxiv.org/abs/1501.07866) is the original paper to analyze the speaker accent recognition dataset . Binary classification is performed in this paper classifying audio samples into US accent or non-US accent (the balanced case)

[2](https://ieeexplore.ieee.org/abstract/document/9259902) is the only work considers the imbalanced case only by comparing the model performance using alternative metrics like MSE, Kappa, precision and recall 

In my work, I tackled the imbalanced case implementing the state-of-the-art imbalaced learning algorithms  

   # Data Exploration 


```python
#importing libs
import numpy as np
import pandas as pd
```


```python
# import dataset
df = pd.read_csv("accent-mfcc-data-1.csv") 
```


```python
index = df.index
columns = df.columns
values = df.values
```


```python
#dimension of dataset
df.shape #or values.shape
```




    (329, 13)




```python
#peek at the data
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>X11</th>
      <th>X12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ES</td>
      <td>7.071476</td>
      <td>-6.512900</td>
      <td>7.650800</td>
      <td>11.150783</td>
      <td>-7.657312</td>
      <td>12.484021</td>
      <td>-11.709772</td>
      <td>3.426596</td>
      <td>1.462715</td>
      <td>-2.812753</td>
      <td>0.866538</td>
      <td>-5.244274</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ES</td>
      <td>10.982967</td>
      <td>-5.157445</td>
      <td>3.952060</td>
      <td>11.529381</td>
      <td>-7.638047</td>
      <td>12.136098</td>
      <td>-12.036247</td>
      <td>3.491943</td>
      <td>0.595441</td>
      <td>-4.508811</td>
      <td>2.332147</td>
      <td>-6.221857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ES</td>
      <td>7.827108</td>
      <td>-5.477472</td>
      <td>7.816257</td>
      <td>9.187592</td>
      <td>-7.172511</td>
      <td>11.715299</td>
      <td>-13.847214</td>
      <td>4.574075</td>
      <td>-1.687559</td>
      <td>-7.204041</td>
      <td>-0.011847</td>
      <td>-6.463144</td>
    </tr>
  </tbody>
</table>
</div>




```python
# list the levels for the class
df['language'].unique()
```




    array(['ES', 'FR', 'GE', 'IT', 'UK', 'US'], dtype=object)




```python
# Grab your wav 
from scipy.io import wavfile
mywav = '../accent-recognition-mfcc--1/ES-M-1-1.wav'
rate, data = wavfile.read(mywav)
print ('Length in time (s): ', np.shape(data)[0]/float(rate))
```

    Length in time (s):  0.7169160997732427
    


```python
# pip install ipython
import IPython.display
IPython.display.Audio(data=data,rate=rate)
```





<audio  controls="controls" >
    <source src="data:audio/wav;base64,UklGRiT3AABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQD3AADo/+b/6P/j/+3/4//v/+H/7f/j/+r/6P/v/+3/7//v/+3/7//q/+b/5v/o/+H/9//m//L/8v/o//z/9P/5/wIA/P8AAPz//P/5//z/+f/3//f/9P/3/+//8v/v//L//v/5/wAA+f/3//f/8v8AAPz/FgARACcAIgA4ACcANQAiABgAFgD3//7/3P/j/9L/3P/h/+j//P/5/xYACQAnAA4AGgACAPz/7//j/9n/1P/X/9T/6P/y/wIAEwAfABoALAAWAB0AFgAEAAkA7f/y/+H/6v/v//L/9P/+//T/DAACAAcADgAAAAQAAAD+/wIAAAACAP7//P/0//n/7/8CAO3/AgD3/wAABwAJABMAEwAOAAwAAgD+//T/8v/m/+H/5v/Z//L/8v/+/w4AAAATAAIACQAAAP7/8v/v/9n/3v/S/9n/4f/e/+r/6v/y//z//P8CAP7/AADq//L/1P/c/83/1//S/+H/4//t//L/9P/+//n/AAD0//f/8v/t/+//4f/m/9T/2f/P/8//1//S/97/1P/Z/97/1P/m/9f/4//X/9f/z//I/9T/wf/X/8H/y//G/8v/xv/S/83/0v/U/9L/1P/Z/8//0v/N/8P/w/+y/7z/t//D/8H/y//D/8j/y//P/9f/1//X/8//z//I/8j/y//B/8v/wf/G/8j/zf/S/9L/1P/P/9T/1P/X/+H/1//e/8v/0v/P/8//3P/S/9f/1//P/9f/2f/j/+b/7f/m/+j/6P/j/+j/6v/h/+P/4f/Z/+H/4//c/+3/6P/q//L/7//t//f/6P/3/+P/7//j//L/4//v/97/7//m//T/7//5//L/9//y//n//P8AAAIA/P8CAPT/AADv//z/7//y//f/7f8AAOj//v/q//f/7//t//L/5v/v/+H/7f/q/+3/+f/0/wAA+f8CAPn/BwD5/wcA/P8CAPn//P/5//n//P/5//7/8v/5/+j/7//y/+r//v/v//7/9P/5//n//P/0//7/7//0//T/8v/0//z/7//3//n/7//+/+r/9P/q/+j/7f/h//L/5v/v/+3/6v/t/97/4//h/+P/5v/q/+j/7//y/+3/+f/q//T/7f/q//T/7f/t/+b/4f/c/+b/4//t/+j/6P/o/+b/5v/m/97/4//h/+j/6v/q/+r/5v/e/+P/2f/m/+P/6P/o/+P/6P/h/+j/5v/c/+r/2f/o/+H/4//m/9n/4f/S/97/2f/e/9z/4//e/+3/5v/y/+j/6v/m/+b/6P/q/+3/6P/t/+P/6v/j//T/8v/0//T/6P/v/+3/8v/5//n/AAD+/wAAAgD3//7/8v/5//L//v/y/wIA9P8AAAIAAAACAAAA9/8AAPf/AgD+/wIABAAHAAIACQACAAwADAAHAAkABAACAAcACQAJABMACQATAAcAEwAHABoADgAiABMAHQAaABMAJAAYACwAKQAsAC4AIgAsACIAMwApADMAIgAnABgAJAAiADEAJwA4ACQANQAsACwAMQAsADEALgAsADMAMQA1ADMANQA1ADMAMwAxADUALgAzADUALgA9ACwAOAAnACwAJAAkACQALAAkADEAIgApACIAJAAfAB8AGgAYABoADgAfABMAHwAfACIAKQApAC4AJAApABMAJwARACcAHQAiACkAJAAxACcALAAiACQAHwAnAB8AIgAaABgAGAAWAB8AJAAnACwAJwAnACwALAAsACkAJwAiACQAIgAdACcAGgAnAB0AHwAfABoAHwAkAB0AMwAYACwAGAAdAB0AGAAiAB8AJwAnAC4AMQAzADEAMQAnAC4ALgAsAD0AJwAxAB8AFgAdAA4AHQAYABoAJAAaACwAJwApADMAKQA1ACwAMQAnACkAGgAiABgAHQAfAB0AIgAfAB8AIgAfACQAHQAfABYADAARAAAAEwAAABMABwAMAA4ACQAOABYAEQAkABYAHQATABgAEQAaABYAGAAWAA4AEQAJAAwAAAAEAP7/BwD+/wkA/v8EAAcABwAJAAwABwACAAkA+f8MAP7/BwACAAQABwAEABMABwAYAAkAEQAOAAcADgD8/wkA+f8JAP7/DgACABEAAAARAAIAEwAJAAkADgAAAA4ABAAHAA4AAgAMAAcABAAHAAQABwAEAAQABwAAAAcABwAHABEACQARAA4AEQAHABEA/v8OAPz/BwACAAcACQAEABEABwAWABYAEwAdABEAFgAJAAkABwAHAAwAAAAMAP7/BwAEAAQADAAHAAkABwARAAcAFgAHAAwADAAAAAkA/v8HAAQACQACAAcABwAAABEA/P8RAPz/CQD8/wcA/P8JAPz/BAD5//z/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4f/o/+P/6P/j/+b/4//m/+P/5v/h/+r/4f/q/+P/4f/h/+b/3v/y/+P/7f/m/+P/5v/j/+r/4//o/+H/4f/e/+P/4//m/+r/5v/j/+3/2f/t/97/5v/m/+P/6v/c/+r/1P/e/9f/1//Z/9z/1P/Z/9f/2f/c/9T/2f/P/9f/zf/Z/8j/4f/G/+H/xv/c/8j/3P/P/+H/1//c/9L/1//S/9T/1//c/9L/2f/D/9T/yP/S/83/zf/G/8j/vP/N/8H/0v/D/8j/y//D/8b/yP++/8v/vv/D/7z/xv+5/8v/vP/L/7z/w//B/8b/y//L/9f/yP/S/8P/yP/L/8//1P/X/9L/z//I/83/yP/U/83/1//P/9L/0v/S/9f/0v/N/8//y//S/83/1P/P/8//y//I/8v/z//P/8//z//P/8v/zf/L/83/zf/N/83/z//P/9T/z//X/8j/1//D/8//yP/N/8//y//S/8b/z//I/83/y//X/8v/3v/S/9f/3P/U/9z/3v/X/+j/1P/h/9L/0v/X/9T/3P/X/9n/2f/Z/9z/3v/h/+H/4f/h/+H/4f/U/9n/z//Z/9f/2f/X/9n/3P/U/+P/1P/e/97/0v/h/9L/3P/X/9T/3v/S/+P/2f/h/+j/2f/t/9z/5v/o/97/6P/Z/97/1P/e/9z/5v/e/+P/3P/U/97/y//h/9f/6P/j/+j/4//e/+H/3P/Z/97/3v/e/97/4//e/+P/5v/h/+r/5v/o/+j/7f/o/+r/6v/e/+//2f/3/+P/8v/j/+j/4//j/+3/6P/3/+//9P/y//L/8v/0//f/+f/y/+//7//t/wAA9P8EAPn/+f/5//T/AAAAAAcABAAEAAAAAAAJAP7/GAAAABYAAgAOAAIADAAHAAQADgACAAwAAgAHAAQABwAMAAcAEwAEABYACQAWABEAGgATABgAEwAOABoAEQAdABoAHwAdACIAGgAdABoAGgAkAB0ALAAdACcAGAAkABoAJAAfAB8AIgAiACcAJAAsACcALAApACcAJAAkACcALAApAC4ALgAsADEAKQAxACkAMQAdAC4AGgAxACcAMQA4ACwAMwAsACcALAAnACwAKQApACQAJwAnACQAMQAnADMAJwAuACkALAAxACkAMwAkADEAJAAxACwAKQApAB8AIgAnACIALgAsACwAMwAuADEANQAxADUAMQA1ACkANQAnADUAMwA4AD0AOgA4ADgAMwA1ADMAPQAzAEwAMQBOADoASQBJAEQATAA/AEQAOgA9ADgAPQA1AD8ANQA9ADoAPQA/AEQARwBCAEcAPQBCAD0AQgBEAD8AQgA/ADoARAA9AEQAPQBJAD8ATABJAEcARwBEAEcASQBYAEwAUwBJAEkASQBMAE4AUwBQAFMATgBTAFUAUABVAEkAWABQAFoAWABaAF0AWgBdAFoAZABfAGQAYgBdAGcAWABpAGIAaQBwAG4AcwBwAGcAawBiAGsAZwBwAGsAbgBuAGcAbgBpAHMAdQB1AHAAcABkAHMAYgB4AGQAcwBpAGsAcABpAGkAbgBnAHAAaQBpAGcAYgBfAF0AYgBdAGkAXQBuAF0AcABYAHUAVQBzAF8AaQBwAGQAcwBpAHAAcwBwAHgAbgBzAGsAcAB4AHUAfQB1AHgAcwB1AHUAeAB4AHgAdQB1AHgAegB6AH0AfQB1AIkAcACOAHgAhAB9AHAAegBrAHoAcwB6AHMAdQBuAHgAcAB4AHAAcwBwAHgAbgB4AHAAegBzAH8AcwB6AG4AcABfAGkAWABpAGQAcwBuAHUAdQBrAHUAcwBpAHgAXwBkAF8AVQBdAF8AWgBkAF0AZABdAGsAYgBrAGcAYgBfAFoAWABdAF0AZABfAGkAWABpAFgAZABaAFoAUABTAEcAVQBJAFUATgBOAEcARwBCAEcASQBOAEkAUwBMAE4ASQBEAD0AOAA1ADEAOgA4ADUAOgAsADMALgAuADgAOgA6AEIALgA1AC4ALAAxAC4ALAAzACcALAAsACwAMwAzADMALAApABYAHwAWAB0AHwAkACcAKQAuAC4ALgApACIAGAAWAAwAFgAJABMACQAJAA4ACQARAAwACQAHAAQA/v/+//f/9//+//z/BwD8/wIA8v/5//T//v8HAAkAGAAOABMABAD8//z/7f/3//T/7//8/+r/+f/3//T//P/q/+//3v/t/97/7//o/+//5v/0/+H/+f/t//z//P/3//n/6v/t/+H/6v/o/+r/6P/Z/9f/1P/X/+j/6v/3//T/9//0//L/9P/o/+H/0v/L/8//zf/j/+b/4//e/9L/zf/c/9T/5v/Z/9n/zf/S/83/3v/e/+j/3v/j/83/4//L/9z/1//L/9z/yP/X/8v/z/++/8v/wf/X/+H/6v/5/+j/7f/X/9T/y//G/7z/sv+w/6j/uf+5/8j/y//U/9n/6v/y//z/AAD5/+//5v/Z/9n/3v/j/+3/6P/h/83/uf+h/5X/kv+c/6b/t/+w/7T/o/+j/6H/uf/B/9z/2f/X/8j/tP+h/5n/kv+V/5L/mf+I/5L/hv+Q/5n/lf+e/5n/lf+e/5L/of+Z/6j/of+3/7D/tP+0/57/nv+G/4H/cv93/2v/bf9o/3X/ev+h/57/yP+t/8H/mf+m/5X/q/+r/7T/of+X/4H/df93/3//jf+c/57/pv+m/6b/qP+e/5z/i/98/3f/Yf9t/1//cP96/3z/jf9//4H/df93/3X/fP93/3//cv+D/3//nv+c/7D/pv+V/4j/cP9t/2T/Zv9h/2T/X/9h/1//XP9h/1//ZP9t/3D/df91/2v/X/9a/1z/Wv9r/2b/XP9a/0v/UP9f/2j/cP9t/1L/Rv83/zX/PP8//0n/S/9Q/1z/ZP9t/23/Vf9J/zr/Lv86/zf/Nf83/xz/Gv8Q/xz/Lv88/0n/P/83/yv/If8r/y7/Nf8w/yH/Ff8G/xD/Cf8Y/w7/EP8J/w7/E/8Y/yH/Dv8B//D+2P7a/tj+6/7//hz/Jv81/yn/JP8Q/w7/C/8Q/xP/EP/4/vP+2P7f/t3+7v7w/vj+8/7z/u7+8P71/vX+9f7i/tD+wv6z/rj+wv7Y/ub++P79/v3+//7z/uL+0P64/qT+p/6u/sn+2P7p/un+7v7u/v3+9f4G//P+6/7d/sf+xP6z/r/+s/7Q/tP+6/74/vX+BP/w/gT/8P79/uT+0/7E/rX+wv7L/ub+BP8O/yb/Gv8Q//X+zv6x/qL+n/6z/rr+x/69/rX+vf61/s7+1f7f/uL+3f7i/t3+6/7r/vX+7v7w/tX+0P66/q7+rP6k/qf+ov6i/pj+qf6u/sv+2v7d/un+3/7r/un+6/7p/tj+3/7T/vX++P4J//D+4v66/r3+vf7T/t/+2P7H/rX+rv7H/tr+Af8B/wT///76/hj/JP81/zf/JP8V/wn/GP8f/0b/Sf9a/1L/UP88/zf/HP8V/wn/Ff8f/0T/Uv9o/2v/bf9f/2b/UP9f/0v/Vf8//0T/MP83/zr/Sf9k/3L/f/96/2T/Wv9J/03/XP9o/23/Zv9a/0v/UP9S/1r/bf9f/2b/Uv9N/0T/UP9V/3f/d/+L/2j/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w//L/8//yP/D/8P/vv/I/8//0v/X/8v/y//D/83/z//h/9z/7f/Z/+r/0v/e/9z/3P/h/97/3P/e/97/4f/q/+j/9P/h//L/4f/y/+3/8v/y/+//7//5//L/AgD0//z/9P/y//n//P8CAAcAAgACAAAA/v8AAAQABAAMAAkACQAOAAIADAD8/wQAAgAEABMADAATAAwADAAHABYADAAiABoAIgAdABoAGgAWAB0AGAAiAB0AHwAfACIAHQApABoAKQAfACIAGgAdABYAFgAWABEAFgARABoAFgAiAB8AJwAfACIAFgAdABYAGgAaAB8AFgAnAA4AKQAWACIAHwAdACQAFgAfABEAGgAdABgAKQAaACQAGgAdAB0AHwAiACkAIgAuABgAKQATABgAGAAWAB0AHwAfAB0AGgATAA4AEwARABgAHQAaABoAGAARABEADgAJAA4ACQAJAA4ADAAYABEAGgARABYAEQARABMAHwAWACQAEQATAA4ADAATABMAGAATABgAEQAWABoAFgAdABoAGAAkABYALgAkADMALgApACQAHQAYABYAFgAYABoAHQAfABgAGAAYABEAHQAWACIAFgApAB8AJAApAB8AJwAdACIAIgApACcALAAfACIAHwAiAC4AJAAxAB0AJwAdACIAIgAnABoAKQAWACkAHQAuACcAMQAsACwALAAnACkALAAuACwAKQAkACIAIgAiABgAJwAdACQAIgAdAB8AGAAWABgAHwAdACcAHwAfABoAEQAWABEAHwATACQADAAfAAcAHQARABYAGAAJABEADgAEABgACQAfABMAHQAYABoAGgAaAB0AGAAaABgAGAAWABgAEwAWABYAFgATABYADAATABEAFgAOABMACQAOAA4ADgAaAAcAFgAAAAkABAAJAAkADgAJAAwABwAHAAQAAAAJAPf/BAD+/wAABwAEAAAABwD+/wAAAgD5/wcA9P8JAPT/DAD8/wkABAAAAAcA/P8AAAAA/P8EAAAACQAAAAQA+f8CAPz/AgD+/wIA/v8AAAcA9/8JAPL/AgD3/wIA/P8CAPf/9//0//n//v8CAAcA/v8CAP7//v8JAAIACQACAAIA/v8CAP7/AgAHAAcABAAMAPz/CQD+/wQAAgARAA4AFgARAAwACQAJAAQADAAOABYAEwAYABEAGAAaAB0AKQAfADUA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o/+m/7n/vv/N/7z/yP+w/8v/t//N/7z/t/+0/6H/sv+t/7n/wf/B/9L/0v/q/+j/9//t/+j/1//S/8v/z//X/97/4f/q/+b/7//h//f/4/8AAPf/AgACAPT/8v/X/9n/yP/e/97/+f8EAAwAEwAOABEACQAEAOr/7f/Z/+b/8v/y/wkA9P8CAO//AADt//T/7//0//T/+f/y//L/+f/5/wcADgAEAAcA9P/3/+b/7f/o/+//9//q/wAA6P/+/+3/+f/y//T/7//t/+//9P/8/wkABwAEAAAA/v8AAAIABAD5/wIA6P/3/+b/7//j/+b/3P/j/+//8v8JAPz/DAAAAAQACQAHAA4A/P8EAPL//P/8//f/+f/v/+r/7//0//z/AgAEAP7/AgD5//n//P/3//7/+f8EAAAACQAEAA4AEQAWAAkACQDy/wAA8v8CAP7/AAAJAPz/EQAAAA4ACQAMAA4AAgACAPf/+f/0/wQAAgAOAAkABAACAPz/9P/q/+r/3v/t/+b/AgD+/xEACQAAAP7/6v/0/+b/7//q/+//7f/0/+3/9P/o/+3/6v/v//T/9//y/+//4//h/9T/2f/S/+H/1//q/97/8v/j/+//6P/o/+r/5v/o/+j/7f/v/+//5v/h/9f/1//h/+H/6v/q/+b/6v/j/+//7/8AAPT//P/t/+b/6v/e//L/7f/3/+//6P/q/+H/7f/v/+r//P/h//f/4f/o/+H/6P/h//L/5v/0/+b/8v/o/+//5v/t/9n/6P/e/+H/5v/m/9n/5v/S/97/4f/h/+r/7f/o/+3/3v/j/97/6P/m/+j/3P/e/8//3P/S/9f/1//N/9f/xv/e/83/6P/Z/+H/4f/Z/+j/2f/m/9n/1//Z/8v/1P/N/9f/0v/U/8v/y//P/8v/1P/N/9L/y//S/9L/2f/X/9n/1P/U/9T/1P/Z/9T/3P/S/9T/z//N/8v/1P/G/97/yP/c/8//1P/X/9T/3v/Z/+b/5v/j/+b/3v/j/97/3v/e/9z/3v/e/97/4//h/+j/3v/t/97/6v/h/+3/3v/3/97/9P/q/+j/9P/q//f/8v/3/+3/+f/q//n/6P/3/+r/9//v//n/8v8AAPL/AADv/wIA8v/+//z/+f/+//T//v/5/wwA8v8JAPf/AAACAP7/BAAJAAcADAAHAAcABAACAAIABwAAAAwAAgAJAAkADgAOAA4ADgARAAcAGAAEABoACQATAA4AEwATABMAEwAWAAwAGAAMABoAEwAdABMAHwATAB8AGgAiACIAHQAfABYAHwATACIAFgAnABgAIgAfABgAKQAdACQAJwAdACkAGAAnABgAJwAaACkAGgAiAB8AHwAiACIAHQAnAB0AJAAkACIAMQAiAC4AIgAnACQAKQAiAC4AIgApACwAJAAzACkAKQAuACIAKQAsACQALAApACQALAApACQAMQAkACwAKQAkACwAKQAkADMAIgAzACQALgAnADEAIgA1ACQAMwApAC4ALgApADEAJAAxAC4ALAAxACkALAApACwALgAsADUAJwAxACkAKQAuACwA" type="audio/wav" />
    Your browser does not support the audio element.
</audio>





```python
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import wave
```


```python
pd.set_option('display.max_colwidth', None)
```


```python
spf = wave.open("../accent-recognition-mfcc--1/ES-M-1-1.wav", "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")

# plt spanish male wave sample
figure(figsize=(18,6), dpi=80)
plt.title("Spanish male sample Signal Wave")
plt.plot(signal)

plt.show()
```


    
![png](output_14_0.png)
    



```python
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,6))

#mfcc spectrogram
from matplotlib import cm
mfcc_data= np.swapaxes(df.drop(['language'], axis=1), 0 ,1)
ax1.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax1.set(xlabel='', ylabel='Ceptrum Indices (X1-X12)', title='MFCCs Spectrogram')
ax1.axes.yaxis.set_ticks([])

ax2.plot(df.iloc[:,1:13])
ax2.set(xlabel='', ylabel='', title='MFCCs Range over Data')
ax2.legend(list(df.iloc[:,1:13].columns), loc='upper right')

plt.show()
```


    
![png](output_15_0.png)
    


As shown below, the collected data has more (US) accent data than all of the other five countries i.e. distribution of examples across the known classes is biased or skewed. **The distribution is neither slight biased nor severe imbalanced**. The majority class is about 5th times each of the other classes.

Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. This results in models that have poor predictive performance for the minority class.


```python
# Class Distribution (imbalance check)
print(df.language.value_counts())
df.language.value_counts().plot(kind='bar', title='Count (target)');
```

    US    165
    UK     45
    IT     30
    FR     30
    GE     30
    ES     29
    Name: language, dtype: int64
    


    
![png](output_17_1.png)
    



```python
print(df.language.value_counts(normalize=True))
```

    US    0.501520
    UK    0.136778
    IT    0.091185
    FR    0.091185
    GE    0.091185
    ES    0.088146
    Name: language, dtype: float64
    

As the plot below shows, the variety of values in the predictors (Cardinality) is high.

Also, no missing Values in our dataset


```python
#checking the cardinality of the columns (count unique values in columns)
df.iloc[:,1:13].nunique().plot.barh(figsize=(12,6))
plt.ylabel('features')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.xlabel('# of unique values')
plt.title('Cardinality')
plt.show()
```


    
![png](output_20_0.png)
    



```python
# cardin = [predictors[col].nunique() for col in predictors.columns.tolist()]
# cols = [col for col in predictors.columns.tolist()]
# d = {k:v for (k, v) in zip(cols,cardin)}
# cardinal = pd.DataFrame(list(d.items()), columns=['column', 'cardinality'])
# cardinal.sort_values('cardinality',ascending=False)
```


```python
"""
 checking missing values 
 that is data=zero in this case because The MFCCs are the amplitudes of the resulting spectrum
  and zero is the only value doesnt make sense in this case

  we know already from UCI page that all data are presented am just following the standard prosudres
"""
print(df.isnull().sum())
#df.columns[df.isnull().sum()>0] index of columns with null
```

    language    0
    X1          0
    X2          0
    X3          0
    X4          0
    X5          0
    X6          0
    X7          0
    X8          0
    X9          0
    X10         0
    X11         0
    X12         0
    dtype: int64
    


```python
#Statistical Summary
df.describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>X11</th>
      <th>X12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.645227</td>
      <td>-4.270814</td>
      <td>2.635319</td>
      <td>7.200207</td>
      <td>-5.648733</td>
      <td>9.810236</td>
      <td>-9.408053</td>
      <td>5.117328</td>
      <td>-1.229432</td>
      <td>-2.362288</td>
      <td>2.430833</td>
      <td>-3.980415</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.105302</td>
      <td>3.514350</td>
      <td>3.635323</td>
      <td>4.310095</td>
      <td>4.596180</td>
      <td>3.624654</td>
      <td>2.484117</td>
      <td>2.650608</td>
      <td>3.634849</td>
      <td>5.042034</td>
      <td>3.478467</td>
      <td>2.985879</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.067831</td>
      <td>-14.972962</td>
      <td>-6.186583</td>
      <td>-8.844231</td>
      <td>-15.656596</td>
      <td>-3.528964</td>
      <td>-15.365665</td>
      <td>-2.873862</td>
      <td>-15.510974</td>
      <td>-11.429178</td>
      <td>-13.664104</td>
      <td>-13.724103</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.065603</td>
      <td>-6.515764</td>
      <td>0.137042</td>
      <td>4.700874</td>
      <td>-8.417684</td>
      <td>8.643563</td>
      <td>-11.120860</td>
      <td>3.482167</td>
      <td>-1.749082</td>
      <td>-5.083522</td>
      <td>1.197789</td>
      <td>-5.941409</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.954206</td>
      <td>-4.252512</td>
      <td>2.029268</td>
      <td>7.804680</td>
      <td>-6.786670</td>
      <td>10.379330</td>
      <td>-9.710399</td>
      <td>4.843103</td>
      <td>-0.389970</td>
      <td>-3.323147</td>
      <td>3.169703</td>
      <td>-4.374334</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.793648</td>
      <td>-1.560250</td>
      <td>4.178026</td>
      <td>10.129165</td>
      <td>-4.463308</td>
      <td>11.784360</td>
      <td>-7.989370</td>
      <td>6.588931</td>
      <td>0.779993</td>
      <td>-1.506037</td>
      <td>4.673452</td>
      <td>-2.191658</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.749851</td>
      <td>3.570765</td>
      <td>17.066487</td>
      <td>16.178942</td>
      <td>7.912809</td>
      <td>21.445837</td>
      <td>-0.424033</td>
      <td>13.846083</td>
      <td>4.789989</td>
      <td>16.326455</td>
      <td>9.166066</td>
      <td>5.259430</td>
    </tr>
  </tbody>
</table>
</div>



The stastical summary, the MFCCs numerical values lies in the window [-16, 21] with similar and low standard deviation is between [2.4 5.1], though **no scaling is needed** before applying classification models


```python
#types of attributes
df.dtypes
# all float
```




    language     object
    X1          float64
    X2          float64
    X3          float64
    X4          float64
    X5          float64
    X6          float64
    X7          float64
    X8          float64
    X9          float64
    X10         float64
    X11         float64
    X12         float64
    dtype: object



All independent variables are numerical only to change dependant variable "language" to categorial 


```python
# change the target datatype in order to encode 
df["language"] = df["language"].astype('category') 
df.dtypes
```




    language    category
    X1           float64
    X2           float64
    X3           float64
    X4           float64
    X5           float64
    X6           float64
    X7           float64
    X8           float64
    X9           float64
    X10          float64
    X11          float64
    X12          float64
    dtype: object




```python
# text coding needed for visualize and process 
language=df['language']

df["language_coded"] = df["language"].cat.codes #add another col holds categories codes (initially will be integer)
df["language_coded"] = df["language_coded"].astype('category')  #(change the col datatype to category)
df=df.drop(['language'], axis=1) # drop the old language col
df.dtypes
```




    X1                 float64
    X2                 float64
    X3                 float64
    X4                 float64
    X5                 float64
    X6                 float64
    X7                 float64
    X8                 float64
    X9                 float64
    X10                float64
    X11                float64
    X12                float64
    language_coded    category
    dtype: object




```python
# list new for the class 
df['language_coded'].unique()
print(df.language_coded.value_counts())
```

    5    165
    4     45
    3     30
    2     30
    1     30
    0     29
    Name: language_coded, dtype: int64
    

['ES', 'FR', 'GE', 'IT', 'UK', 'US'] = [0, 1, 2, 3, 4, 5]


```python
predictors=df.iloc[:,:-1] #df.iloc[:,0:12]
target=df.iloc[:,-1:] #df.iloc[:,12:13]    =#target = 'language_coded'
```

# Data Distribution

**Many classification algorithms, assumes variables to follow a particular distribution (specially Gaussian distribution)**. The cost of not meeting the assumptions could be high at times.

In statistics, **D'Agostino-Pearson's** test, is a goodness-of-fit measure of departure from normality, that is the test aims to establish whether or not the given sample comes from a normally distributed population. The test is based on transformations of the sample kurtosis and skewness, and has power only against the alternatives that the distribution is skewed and/or kurtic.

Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in the distribution.

Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used statistical test for normality.

According to the D’Agostino-Pearson test, the data is normally distributed when the test statistic $Z^2_K+Z^2_S$ has a chi-square distribution with 2 degrees of freedom


```python
'''
  
  stats.normaltest function tests the null hypothesis that a sample comes from a 
  normal distribution It is based on D’Agostino and Pearson’s test that combines 
  skew and kurtosis to produce an omnibus test of normality
  
  Usually, a significance level (denoted as α or alpha) of 0.05 (standard) indicates 
  that the risk of concluding the data do not follow a normal distribution—when, 
  actually, the data do follow a normal distribution—is 5%.

'''

from scipy import stats
for i in range(len(df.columns)-1):
    k2, p = stats.normaltest(df.iloc[:,i:i+1])
    alpha = 0.05 #1e-3
#     print("p = {}".format(p))
    print(df.columns[i])
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected") #not normal
    else:
        print("The null hypothesis cannot be rejected") #normal
```

    X1
    The null hypothesis can be rejected
    X2
    The null hypothesis cannot be rejected
    X3
    The null hypothesis can be rejected
    X4
    The null hypothesis can be rejected
    X5
    The null hypothesis can be rejected
    X6
    The null hypothesis can be rejected
    X7
    The null hypothesis can be rejected
    X8
    The null hypothesis can be rejected
    X9
    The null hypothesis can be rejected
    X10
    The null hypothesis can be rejected
    X11
    The null hypothesis can be rejected
    X12
    The null hypothesis can be rejected
    

**Only X2 comes from normal distribution for a $\rho$ of 5%**


```python
from scipy.stats import norm
import seaborn as sns

plt.figure(figsize=(22,30))
try:
    for i, col in enumerate(predictors.columns.to_list()):
        plt.subplot(4, 3, i + 1)
        sns.distplot(df[col], bins=15, fit=norm, label=col)        
        plt.legend()
        plt.title(col)
        plt.tight_layout()
except Exception as e:
    print(col,e)
```


    
![png](output_36_0.png)
    


the figure above shows the plots of the predictors distribution with a histogram and maximum likelihood gaussian distribution fit.

- Only X2 comes close to a normal distribution (confirms the D'Agostino-Pearson's test results)
- X3,X5,X9,X10,X11 have highly skewed distribution, and X4,X6 with moderate Skew distribution


```python
print(predictors.skew())
```

    X1     0.192326
    X2    -0.226634
    X3     1.131745
    X4    -0.743209
    X5     1.180070
    X6    -0.985631
    X7     0.483211
    X8     0.394249
    X9    -1.637664
    X10    1.359030
    X11   -1.686876
    X12    0.321322
    dtype: float64
    

# Outliers Detection

Outliers are innocent until proven guilty. With that being said, they should not be removed unless there is a good reason for that.
For example, one can notice some weird, suspicious values that are unlikely to happen, and so decides to remove them. Though, they worth investigating before removing.
It is also worth mentioning that some models, like Support Vector Machine, are very sensitive to outliers. In other words, outliers might throw the model off from where most of the data lie.

we will examine two outlier detection techniques and later will compare generated noise-free dataset split against other splits

   ## Interquartile Range (IQR)
   
A boxplot can be used to indicate explicitly the presence of outliers. Many inferential procedures are based on the assumption that the population distribution is normal. Even a single extreme outlier/several mild outliers in the sample warns the investigator that such procedures may be unreliable

In the Interquartile Range (IQR) technique, Any observation farther than 1.5fs from the closest fourth is an outlier. An outlier is extreme if it is more than 3fs from the nearest fourth, and it is mild otherwise.


```python
fig = plt.figure(figsize=(15,10))
fig.suptitle('Boxplot 1.5*IQR', fontsize=14)

ax=sns.boxplot(data=predictors)
ax=sns.swarmplot(data=predictors, color=".25", size=3)

plt.tight_layout()
```


    
![png](output_41_0.png)
    


As we can see from the plot, we notice that X3,X5,X6,X9,X10 and X11 are stacked with many outliers outside the 1.5*IQR 


```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
#fact: quantile works on numeric dtypes so df.quantile=predictors.quantile
```

    X1     7.728045
    X2     4.955514
    X3     4.040984
    X4     5.428291
    X5     3.954376
    X6     3.140798
    X7     3.131490
    X8     3.106763
    X9     2.529075
    X10    3.577485
    X11    3.475663
    X12    3.749751
    dtype: float64
    


```python
def outliers_det(df_,th):
    
    df_2cln = df_.copy()
    predictors_=df_2cln.iloc[:,:-1] 
    Q1 = df_2cln.quantile(0.25)
    Q3 = df_2cln.quantile(0.75)
    IQR = Q3 - Q1
    
    #filter out the outliers by keeping only valid values
    # returns True/False list of rows (329) where false=outlier  (.any changes numbers to boolean)
    inliners=~((predictors_< (Q1 - th * IQR)) |(predictors_> (Q3 + th * IQR))).any(axis=1)

    outliers = []
    for i in range(len(df_2cln)):
        if (inliners[i] == False):
            outliers.append(i)  # index of the outlier

    outliers_idx=np.array(outliers)

    noise_df = df_2cln.index.isin(outliers_idx)
    cln_df = df[inliners] #data split#1
    print('noise-free data split created with length = ',len(cln_df))
    print('outliers = ',len(df_2cln)-len(cln_df))
    print("data filtered percentage ="+"{:.2f}" .format((len(df_2cln)-len(cln_df))/len(df_2cln)))
    return outliers_idx,cln_df,noise_df
```


```python
outliers_indices_iqr,noisefree_iqr_df,outliers_iqr_df = outliers_det(df,1.5)#1.5 cuts more data & given horrible results on classification (trial and error)
```

    noise-free data split created with length =  261
    outliers =  68
    data filtered percentage =0.21
    

The filtered data (outliers) percentage according to 1.5*IQR is 21% of the whole data set

That's a large amount considering that the dataset is small for a 6 class classification 

## Mahalanabois Distance
The Mahalanobis distance between an observation $(x_i)$  and the dataset mean ($\hat{\mu}$) given by:

\begin{equation}
d^2_{\hat{\mu},\hat{\Sigma}}(x_i) =
(x_i-\hat{\mu})^T\hat{\Sigma}^{-1}(x_i-\hat{\mu})
\end{equation}

 $\hat{\mu}, \hat{\Sigma} $ being respectively estimates of the dataset mean and covariance., where $\Sigma$ is a $d\times d$ covariance matrix.

The sample mean and covariance matrix can be quite sensitive to outliers,and that's the reason why Mahalanobis distance is effective on multivariate data is because it uses covariance between variables in order to find the distance of two points. In other words, Mahalanobis calculates the distance between point “P1” and point “P2” by considering standard deviation (how many standard deviations P1 far from P2)

Then, Multivariate outliers can be simply defined as observations having a large squared Mahalanobis distance.


```python
def outliers_maha(df):

    df1=np.array(df)
    cov_matrix = np.cov(df1, rowvar=False) 
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    vars_mean = []
    for i in range(df1.shape[0]):
        vars_mean.append(list(df1.mean(axis=0))) 
    diff = df1 - vars_mean

    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))

    std = np.std(md)
    k = 2. * std # k = 3. * std if extreme else 2. * std
    m = np.mean(md)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(md)):
        if (md[i] >= up_t) or (md[i] <= low_t):
            outliers.append(i)  # index of the outlier

    indices_=np.array(outliers)
    outliers_df = df.index.isin(indices_)

    cln_df = df[~outliers_df]

    print('\nnoise free data split created with length = ',len(cln_df))
    print('cleared data = ',len(df)-len(cln_df))
    print("data filtered percentage ="+"{:.2f}" .format((len(df)-len(cln_df))/len(df)))
    
    return indices_,cln_df,outliers_df
```


```python
outliers_indices_maha, inliers_maha_df,outliers_maha_df=outliers_maha(df)
```

    
    noise free data split created with length =  318
    cleared data =  11
    data filtered percentage =0.03
    


```python
# print(df.index.isin(outliers_indices)) index of outliers
# print(language[outliers_iqr_df].value_counts(sort=False)[::-1]);
# print(language[outliers_maha_df].value_counts(sort=False)[::-1]);
```


```python
fig = plt.figure(figsize=(10,6))

maha_counts=language[outliers_maha_df].value_counts(sort=False)[::-1]
iqr_counts=language[outliers_iqr_df].value_counts(sort=False)[::-1]
  
X_axis = np.arange(len(language.unique()))
  
plt.bar(X_axis + 0.2, maha_counts, 0.4, label = 'Mahanapolis')
plt.bar(X_axis - 0.2, iqr_counts, 0.4, label = 'IQR')
  
plt.xticks(X_axis, language.unique()[::-1])
plt.xlabel("")
plt.ylabel("# of Outliers")
plt.title("Outliers class distribution")
plt.legend()
plt.show()
```


    
![png](output_51_0.png)
    


That is the **outliers class distribution**, and as we can see, both outliers detectors affected by the class imbalance i.e. bias towards the majority class

Outlier Removal Tradeoff
-------------------------------------
- we might run the risk of information loss which will cause our models to have a lower accuracy specially with the limited data in hand
- both outliers detectors affected by the class imbalance i.e. bias towards the majority class
- According to UCI, the collected data is noise free

<!-- actually play with threshold and see how it affects the accuracy of our classification models -->

for the abovementioned reasons, only a dataset split by removing shared outliers between IQR and mahanapolis will be examined for classification 


```python
print("Mahalanabois Outliers Indices: {}\n".format(outliers_indices_maha))
```

    Mahalanabois Outliers Indices: [124 136 224 226 229 231 232 286 287 295 322]
    
    


```python
print("IQR Outliers Indices: {}\n".format(outliers_indices_iqr))
```

    IQR Outliers Indices: [ 14  22  27 157 162 171 194 204 213 215 219 221 224 225 226 227 228 229
     230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247
     248 249 250 251 252 253 263 284 285 286 287 291 292 293 294 295 298 314
     315 316 317 318 319 320 321 322 323 324 325 326 327 328]
    
    


```python
def commonElements(arr): 
    # initialize result with first array as a set 
    result = set(arr[0]) 
    for currSet in arr[1:]: 
        result.intersection_update(currSet) 
  
    return list(result) 
```


```python
outliers_indices=[outliers_indices_iqr,outliers_indices_maha,np.asarray(index).flatten()]
print("Counter of common outliers between IQR & Mahalanobis:", len(commonElements(outliers_indices)))
```

    Counter of common outliers between IQR & Mahalanobis: 9
    


```python
noise = df.index.isin(commonElements(outliers_indices))
noisefree_df = df[~noise]
print('Noise-free dataset split of Shape:{} is created'.format(noisefree_df.shape))
#         print('Length of X (train): {} | Length of y (train): {}'.format(len(X_train), len(y_train)))

#note that indexes of >320 still appears because idx is related to rows and intermediate idx deleted==> idx.shap=261
```

    Noise-free dataset split of Shape:(320, 13) is created
    

# Multicollinearity Detection

Collinearity is a linear association between two variables. Multicollinearity is a situation where the correlations among subsets of the variables

Multicollinearity is not a big issue in classification problems since it is not affecting the prediction power, rather we use it for feature selection to select subset of the most relevant features that helps in classification

Two methods of detection will be examined against the dataset

## Pearson correlation

Pearson correlation coefficient (PCC) is a measure of linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations; thus it is essentially a normalized measurement of the covariance, such that the result always has a value between −1 and 1. As with covariance itself, the measure can only reflect a linear correlation of variables, and ignores many other types of relationship or correlation.


Given a pair of random variables $(X,Y)$ the formula for Pearson correlation coefficient $\rho$ is:

\begin{equation}
\label{eq:dotp}
\rho_{X,Y} = \frac {cov(X,Y)} {\sigma_X \sigma_Y}, 
\end{equation}
where $\sigma$ is the standard deviation ...

In general, **an absolute correlation coefficient of >0.7 among two or more predictors indicates the presence of multicollinearity**


```python
#class to return the matrix of p-value of predictors 
#The P-value is the probability that you would have found the current result if the correlation coefficient were in 
#fact zero (null hypothesis). If this probability is lower than the conventional 5% (P<0.05) the 
#correlation coefficient is called statistically significant
from scipy import stats
def corr_sig(predictors=None):
    p_matrix = np.zeros(shape=(predictors.shape[1],predictors.shape[1]))
    for col in predictors.columns:
        for col2 in predictors.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(predictors[col],predictors[col2])
            p_matrix[predictors.columns.to_list().index(col),predictors.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
# Corr.Plot of Example Data with only Sig. Correlations (plotting with only significant p-value correlation (alpha < .05))

p_values = corr_sig(predictors)

mask = np.invert(np.tril(p_values<0.05)) #0.000000000000000000000000001

#Compute pairwise correlation of columns .corr() default method=pearson
#obviously another way is to return the matrix of _ in the above class
corrmat = predictors.corr() 
top_corr_predictors= corrmat.index
plt.figure(figsize=(7,5))

g=sns.heatmap(df[top_corr_predictors].corr(),annot=True,cmap=sns.diverging_palette(25, 250, s=75, l=60,n=200), mask=mask)
plt.tight_layout()
```


    
![png](output_62_0.png)
    


considering  an absolute correlation coefficient of >0.7 from the heatmap following strong correlations found:

- X5,X6 with X9
- X4,X5,X6 with X10
- X9 with X10

Apparently, will lose info if X9 or X10 is droppped since they go together, so the right decision is to drop X4,X5,X6 

## Hieracichal clustering

It's a type of clustering requires the user to specify a measure of dissimilarity between (disjoint) groups of observations, based on the pairwise dissimilarities among the observations in the two groups.
As the name suggests, they produce hierarchical representations in which the clusters at each level of the hierarchy are created by merging clusters at the next lower level. At the lowest level, each cluster contains a single observation. At the highest level there is only one cluster containing all of the data.

Strategies for hierarchical clustering divide into two basic paradigms: agglomerative (bottom-up) and divisive (top-down). 

**Agglomerative** strategies start at the bottom and at each level recursively merge a selected pair of clusters into a single cluster. This produces a grouping at the next higher level with one less cluster. The pair chosen for merging consist of the two groups with the smallest intergroup dissimilarity

Most agglomerative and some divisive methods (when viewed bottomup) possess a monotonicity property. That is, the dissimilarity between merged clusters is monotone increasing with the level of the merger. Thus the binary tree can be plotted so that the height of each node is proportional to the value of the intergroup dissimilarity between its two daughters. The terminal nodes representing individual observations are all plotted at zero height. This type of graphical display is called a **dendrogram**.

**Figure shows agglomerative clustering and dendrogram with average linkage**


```python
g = sns.clustermap(predictors, row_cluster= False,  metric="correlation")

```


    
![png](output_65_0.png)
    


'X4','X5','X6' can be removed according to Pearson correlation

'X4','X5','X6','X7' can be removed according to dendogram

we use trial and error to choose the right combination gives the best classification results


```python
#trial and error
# @knn (k=3)
# df_colfree=df.drop(columns=['X4','X5','X6','X7'],axis=1) #bad

#all three comp 
# df_colfree=df.drop(columns=['X4','X5','X6'],axis=1)#0.6752641623609366
# df_colfree=df.drop(columns=['X4','X5','X7'],axis=1)#F1 score on test set: 0.7674608993157381
# df_colfree=df.drop(columns=['X4','X6','X7'],axis=1)#F1 score on test set: 0.7898923444976076
# df_colfree=df.drop(columns=['X5','X6','X7'],axis=1)#0.7070455550781195

#two only
df_colfree=df.drop(columns=['X6','X7']) #0.8500541125541126
```

# Dimensionality Reduction

Principal component analysis (PCA) is is unsupervised learning that uses an orthogonal transformation to convert the variables of a dataset into a new set of variables which are linearly uncorrelated. The principal components are ranked according to the variance of data along them. 


\begin{definition}\
  Given observations $x_i^{(k)}$ of random variables $X^{(k)}$, want
  to find <span class="mark">linearly uncorrelated</span> principal components.

  - Write $X = (T{x}_1 | \cdots | T{x}_\ell) \in RR^{N \times \ell}$.
  
  - Calculate Singular Value Decomposition $X = U S V^t$, 
  
  - Then the principal components are the variables \[ Y^{(j)} = \sum_k U_{kj} X^{(k)}. \]

\end{definition}

Most of the variance is captured by $Y^{(1)}$; second to most is captured by $Y^{(2)}$; and so on

**For the above mentioned procedure to be correct, data need to be centered and scaled**

This technique can be used to reduce the dimensionality of the dataset by considering just the most important principal components and wether it will help classifing the accent recognition dataset i.e. if the variance help in seperating classes

sklearn.decomposition.PCA class uses Singular Value Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.



```python
from sklearn.decomposition import PCA

n_components=5
pca = PCA(n_components, random_state=42)

#the following statemend can be done through
#from sklearn.preprocessing import scale, normalize
#norm_predictors = normalize(predictors)

norm_predictors = (predictors - predictors.mean()) / predictors.std()
#pca.fit_transform does not normalize data automatically

X_pca = pca.fit_transform(norm_predictors)
X_pca = pd.DataFrame(X_pca, columns=["PC%d" % (x + 1) for x in range(n_components)]) 

X_pca['language_coded']=df.iloc[:,12:13]
```


```python
print("Sum of explained variance ratio using {} PCA components: ".format(n_components),round(sum(pca.explained_variance_ratio_),2))

print("\nexplained variance ratio by each PCA component")
print(pca.explained_variance_ratio_)
```

    Sum of explained variance ratio using 5 PCA components:  0.86
    
    explained variance ratio by each PCA component
    [0.48029376 0.16148163 0.09600608 0.0736275  0.04723642]
    

- 4 Principal components are enough to explain about 79% 
- 5 Principal components are enough to explain about 86% 


```python
plt.figure(figsize=(10,7))
plt.plot([1,2,3,4,5], pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot([1,2,3,4,5], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')
#1,2,3,4 to avoid PCA1 is at 0 in xscale

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.8,5.2)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4,5])
plt.legend();
```


    
![png](output_73_0.png)
    


Let us see the effect of original predictors on each principal components


```python
#Effect of original predictors (variables) on each components
#too dark = too positive effect , too light= too negative effect
#to do:just abs() coz all we need now is the effect whether its + or -
plt.figure(figsize=(10,7))
ax = sns.heatmap(pca.components_, 
                 linewidths=0.5,
                 annot=True,
                 cmap='Blues',
                 yticklabels=[ "PCA"+str(v) for v in range(1,pca.n_components_+1)],
                 xticklabels=list(predictors.columns),
                 cbar_kws={"orientation": "horizontal"})
plt.setp(ax.get_yticklabels(), rotation=45)
ax.set_aspect("equal")
```


    
![png](output_75_0.png)
    


PCA1 is increased with X5 and X10 and decreased with X4,X6,X9. This suggests that these five coeffiecient goes together, If one increases, then the remaining ones tend to increase/decrease. So this component PCA1 can be viewed as a measure of the quality of X4,X5,X6,X9 and X10

PCA1 is more affected by X9,X10 than X4,X5,X6 (the same result we reached in the multicollinearity section)

# Classification 

## Imbalanced Dataset Classification 

The machine-learning literature on this topic has essentially developed two common solution strategies:
1.	Restore balance on the training set by **undersampling** the large class or by **oversampling** the small class, to prevent bias from arising in the first place.
2.  modify the **costs of misclassification** to prevent bias by penalizing learning algorithms that increase the cost of classification mistakes on the minority class. this technique is implemented in python using the argument **class_weight=’balanced’** within classifier models during training to penalize mistakes on the minority class by an amount proportional to how under-represented it is.We also want to include the argument **probability=True** if we want to enable probability estimates for SVM algorithms.

## Classification Metric

In learning imbalanced data, the overall classification accuracy is often not an appropriate measure of performance.  A trivial classifier that predicts every case as the majority class can still achieve very high accuracy. 

Metrics such as true negative rate, true positive rate, precision, recall, and F-measure are commonly used 

Since there is no distinction between whether Precision is more important or Recall in the accent recognition case, we combine them

**F1-measure** is the harmonic mean of precision and recall:

In practice, when we try to increase the precision of our model, the recall goes down and vice-versa. The F1-score captures both the trends in a single value.

\begin{equation}
F_1score = 2 \left[ \frac {Precision \times Recall} {Precision+ Recall} \right]
\end{equation}

In the multi-class case, there are three ways to generalize F1 scores:
- **macro-averaged** F1, and it weighs each class equally.
- **micro-averaged** F1, and it weighs each sample equally.
- **Weighted-averaged**  each classes’s contribution to the average is weighted by its size

To give equal weight to each class, use macro-averaging

## K-fold Cross-Validation

for estimating prediction error **K-fold cross-validation** uses part of the available data to fit the model, and a different part to test it. We split the data into K roughly equal-sized parts and for the kth part (third above), we fit the model to the other K −1 parts
of the data, and calculate the prediction error of the fitted model when
predicting the kth part of the data. We do this for k = 1, 2,...,K and
combine the K estimates of prediction error.

Let $κ$ : {1,...,N} → {1,...,K} be an indexing function that indicates the partition to which observation $i$ is allocated by
the randomization. Denote by $f^{−k}(x)$ the fitted function, computed with the $kth$ part of the data removed. Then the cross-validation estimate of prediction error is
\begin{equation}\label{eq:}
CV(f)= \sum_{i=1}^{N} L(y_i,f^{-k(i)}(x_i))
\end{equation}

Typical choices of K are 5 or 10 

sklearn package provides the class **StratifiedKFold**; a cross-validation object returns stratified folds. The folds are made by preserving the percentage of samples for each class as shown below


<img src="images/3.png" alt="Drawing" style="width: 500px;" />

## Classification Models

The following classification models gives competent results on the speaker accent recognition dataset: 
### K-Nearest Neighbour:
Nearest-neighbor methods use those observations in the training set $T$ closest in input space to $x$ to form $\hat{Y}$ . Specifically, the k-nearest neighbor fit for $\hat{Y}$ is defined as follows:

\[\hat{Y}(x)=\frac{1}{k}  \sum_{{x_i}\in {N_k (x)}} y_i\]

where $N_k(x)$ is the neighborhood of $x$ defined by the $k$ closest points $x_i$ in the training sample. Closeness implies a metric, Euclidean distance is the default choice in sklearn nearest neighbour algorithm. 
So, in words, we find the $k$ observations with $x_i$ closest to $x$ in input space, and average their responses.

thae k-nearest-neighbor fits have a single parameter, the number of neighbors k, as k increases bias increases and variance decreases and so a small value of k could lead to **overfitting** as well as a big value of k can lead to **underfitting** 

### Support Vector Machine:

When the classes are not separable by a linear boundary, the support vector machine (SVM) classifier maps sample $x_i$ into a feature space of higher dimensions $\phi (x)$ in which the classes can be linearly separated. This results in a non-linear decision boundary in the original dimensions.

As the vectors $x_i$ appear as inner products in the optimization problem to find the saperating hyperplane, the mapping function $\phi(x)$ does not need to be explicitly specified, rather requires only knowledge of the kernel function:

\[K(x_1,x_2)=\phi(x_1)^Tϕ(x_2)\]

The effective hyperparameters of a SVM to be tuned and implemented using sklearn library are:
In the hyperparameter tuning we consider the following two kernels:

1- the kerenel function and the functions chosen are the following:
- linear kernel:               $K(x_1,x_2)= x_1⋅x_2$
- Radial Basis Function (RBF): $K(x_1,x_2)=exp (−\gamma∥x_1−x_2∥^2)$

2- C: the cost parameter, a large value of C will discourage any positive slack variables and lead to an overfit wiggly boundary in the original feature space; a small value of C will causes $f(x)$ and hence the boundary to be smoother.

3- Gamma (RBF parameter): as parameter C, The higher the gamma the more chance of overfitting

### Random Forest
 An average of $B$ i.i.d. random variables, each with variance $\sigma^2$, has variance $\frac{1}{B} \sigma^2$. If the variables are simply i.d. (identically distributed, but not necessarily independent) with positive pairwise correlation $\rho$, the variance of the average to be optimized is

\[\rho \sigma^2+\frac{1-\rho}{B} \sigma^2\]

The idea in **random forests** is to improve the variance reduction of bagging by reducing the correlation between the trees, without increasing the variance too much. This is achieved in the tree-growing process (to the bootstrapped data) through random selection of the input variables $(m =<p)$(with selecting best variables at each step as candidate for splitting).
When used for classification, a random forest obtains a class vote from each tree, and then classifies using majority vote

For classification, the default value for m is $\sqrt{p}$ and the minimum node size is one.

Random forest parameters to be considered in implementation using sklearn library:
- Number of trees in random forest
- Number of features to consider at every split
- Maximum number of levels in tree
- Minimum number of samples required to split a node
- Minimum number of samples required at each leaf node
- Method of selecting samples for training each tree (bootstrap or not)

**Note: Linear Discriminant Analysis and Logistic Regression are used along the abovementioned classifier, but since both gave poor results, they are excluded from the report. Same goes for evaluation metrics precision and recall**

###  Implementation Framework

<img src="images/4.3.png" alt="Drawing" style="width: 600px;" />

The figure above shows the classification framework for the imbalanced and balanced cases. Stratified **10-folds** is performed for cross-validation 


```python
from sklearn.model_selection import train_test_split as tts

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score,KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time as time
```


```python
test_size=0.3
SEED=20 #to correcty compare results of the different datasets splits use the same random_state seed (to pick same instances for df,pca,colfree) only noisefree can't pick the same indices
```


```python
train_data, test_data, train_labels, test_labels = tts(predictors,target, test_size=test_size, stratify=target, random_state=SEED)
train_colfree, test_colfree, colfree_train_labels, colfree_test_labels = tts(df_colfree.drop('language_coded', axis=1), df_colfree['language_coded'], test_size=test_size, stratify=df_colfree['language_coded'], random_state=SEED)
train_noisefree, test_noisefree, trainlabels_noisefree, testlabels_noisefree = tts(noisefree_df.iloc[:,0:12], noisefree_df.iloc[:,12:13], test_size=test_size, stratify=noisefree_df.iloc[:,12:13], random_state=SEED)
```


```python
#prepare pca-transformed dataset for classification
norm_predictors = (train_data - train_data.mean()) / train_data.std()
pca_train = pca.fit_transform(norm_predictors)
norm_predictors = (test_data - test_data.mean()) / test_data.std()
pca_test = pca.fit_transform(norm_predictors)
```


```python
datasets={
              'Original dataset':[train_data,train_labels,test_data,test_labels],
              'PCA':[pca_train,train_labels, pca_test,test_labels],
              'noise-free':[train_noisefree, trainlabels_noisefree, test_noisefree, testlabels_noisefree],
              'coll-free':[train_colfree, colfree_train_labels, test_colfree,colfree_test_labels]
}
```

###  Imbalanced dataset classification


```python
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] #list(range(10, 111, 10)) + [None]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criterion = ['gini','entropy']
# random_state int, RandomState instance, default=None

rf_params = {  'n_estimators': n_estimators,
               'criterion' : criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

knears_params = {"n_neighbors": list(range(3,7,2))}

log_reg_params = {
                  "penalty": ['l1', 'l2'], 
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "solver":['sag' ,'saga', 'liblinear','newton-cg', 'lbfgs'] 
                 }
svc_params = {
              'C': np.logspace(1, 3, 3), 
              'kernel': ['rbf', 'linear'],
              'gamma':np.logspace(-3, -1, 3)
             }

lda_params={'solver':['lsqr','eigen','svd']}
```


```python
params = {
        '1':knears_params,
        '2':log_reg_params,
        '3':svc_params,
        '4':rf_params,
        '5':lda_params,
         }
```


```python
#basline classifiers
classifiers = {
        "KNearest": KNeighborsClassifier(),       
        "LogisiticRegression": LogisticRegression(),
        "Support Vector Classifier": SVC(),         #  (probability=True) is compatible with imbalanced dataset 
        "Rforest":RandomForestClassifier(),        #random_state= 0
        "LDA":LinearDiscriminantAnalysis(),
              }    
```


```python
# cv parameters
splits=10
cv_ = StratifiedKFold(n_splits=splits, random_state=SEED, shuffle=True)
```


```python
def imbalanced_clf (cv=cv_,ds=datasets,clf=classifiers,par=params):
    df_results = pd.DataFrame()
    for key in ds:
        X_train=ds[key][0]
        y_train=ds[key][1]
        X_test=ds[key][2]
        y_test=ds[key][3]

        accuracy_lst = []
        precision_lst = []
        recall_lst = []
        f1_lst = []
        auc_lst = []
        
        for (key1,classifier),(key2,parameter) in zip(clf.items(),par.items()):
            start_time = time.time()
            grid=GridSearchCV(classifier, parameter,cv=cv,scoring='f1_weighted')
            grid.fit(X_train, y_train)
            best_est = grid.best_estimator_
            prediction = best_est.predict(X_train)
            
            training_score = cross_val_score(best_est, X_train, y_train, scoring='f1_weighted',cv=cv)
            row = {
                'Dataset split': key,
                'model':classifier.__class__.__name__,
                'run_time': format(round((time.time() - start_time)/60,2)),
                'F1 CV score': round(training_score.mean(), 2),
                'F1 score on test set': round(f1_score(y_test, best_est.predict(X_test), average='weighted'),4),
                'Precision score on test set': round(precision_score(y_test, best_est.predict(X_test), average='weighted'),4),
                'Recall score on test set': round(recall_score(y_test, best_est.predict(X_test), average='weighted'),4),
                'best parameters':grid.best_params_,
                   }
            df_results = df_results.append(row, ignore_index=True) 
    return df_results
```


```python
imbalanced_clf().sort_values(by='F1 score on test set')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset split</th>
      <th>F1 CV score</th>
      <th>F1 score on test set</th>
      <th>Precision score on test set</th>
      <th>Recall score on test set</th>
      <th>best parameters</th>
      <th>model</th>
      <th>run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>PCA</td>
      <td>0.55</td>
      <td>0.4253</td>
      <td>0.3931</td>
      <td>0.4747</td>
      <td>{'bootstrap': False, 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}</td>
      <td>RandomForestClassifier</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PCA</td>
      <td>0.47</td>
      <td>0.4257</td>
      <td>0.4127</td>
      <td>0.5253</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
      <td>SVC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PCA</td>
      <td>0.44</td>
      <td>0.4450</td>
      <td>0.4705</td>
      <td>0.5051</td>
      <td>{'solver': 'lsqr'}</td>
      <td>LinearDiscriminantAnalysis</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PCA</td>
      <td>0.50</td>
      <td>0.4717</td>
      <td>0.5478</td>
      <td>0.5051</td>
      <td>{'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}</td>
      <td>LogisticRegression</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PCA</td>
      <td>0.57</td>
      <td>0.5244</td>
      <td>0.5242</td>
      <td>0.5253</td>
      <td>{'n_neighbors': 5}</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>noise-free</td>
      <td>0.67</td>
      <td>0.6723</td>
      <td>0.6707</td>
      <td>0.6875</td>
      <td>{'solver': 'lsqr'}</td>
      <td>LinearDiscriminantAnalysis</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>coll-free</td>
      <td>0.60</td>
      <td>0.7086</td>
      <td>0.7206</td>
      <td>0.7172</td>
      <td>{'solver': 'lsqr'}</td>
      <td>LinearDiscriminantAnalysis</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>noise-free</td>
      <td>0.69</td>
      <td>0.7363</td>
      <td>0.7420</td>
      <td>0.7396</td>
      <td>{'C': 1, 'penalty': 'l2', 'solver': 'newton-cg'}</td>
      <td>LogisticRegression</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Original dataset</td>
      <td>0.67</td>
      <td>0.7446</td>
      <td>0.7787</td>
      <td>0.7475</td>
      <td>{'solver': 'lsqr'}</td>
      <td>LinearDiscriminantAnalysis</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Original dataset</td>
      <td>0.69</td>
      <td>0.7489</td>
      <td>0.7589</td>
      <td>0.7475</td>
      <td>{'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}</td>
      <td>LogisticRegression</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>16</th>
      <td>coll-free</td>
      <td>0.65</td>
      <td>0.7598</td>
      <td>0.7687</td>
      <td>0.7677</td>
      <td>{'C': 1, 'penalty': 'l2', 'solver': 'newton-cg'}</td>
      <td>LogisticRegression</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>18</th>
      <td>coll-free</td>
      <td>0.74</td>
      <td>0.7815</td>
      <td>0.7933</td>
      <td>0.7879</td>
      <td>{'bootstrap': False, 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}</td>
      <td>RandomForestClassifier</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>15</th>
      <td>coll-free</td>
      <td>0.75</td>
      <td>0.7904</td>
      <td>0.8002</td>
      <td>0.7879</td>
      <td>{'n_neighbors': 3}</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Original dataset</td>
      <td>0.84</td>
      <td>0.8015</td>
      <td>0.8073</td>
      <td>0.8081</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
      <td>SVC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Original dataset</td>
      <td>0.79</td>
      <td>0.8030</td>
      <td>0.8237</td>
      <td>0.8081</td>
      <td>{'bootstrap': False, 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}</td>
      <td>RandomForestClassifier</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Original dataset</td>
      <td>0.80</td>
      <td>0.8102</td>
      <td>0.8195</td>
      <td>0.8081</td>
      <td>{'n_neighbors': 3}</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise-free</td>
      <td>0.82</td>
      <td>0.8282</td>
      <td>0.8294</td>
      <td>0.8333</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
      <td>SVC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise-free</td>
      <td>0.77</td>
      <td>0.8337</td>
      <td>0.8449</td>
      <td>0.8333</td>
      <td>{'n_neighbors': 3}</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>coll-free</td>
      <td>0.79</td>
      <td>0.8380</td>
      <td>0.8535</td>
      <td>0.8384</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
      <td>SVC</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>noise-free</td>
      <td>0.72</td>
      <td>0.8581</td>
      <td>0.8826</td>
      <td>0.8646</td>
      <td>{'bootstrap': False, 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}</td>
      <td>RandomForestClassifier</td>
      <td>0.26</td>
    </tr>
  </tbody>
</table>
</div>



Results for the imbalanced case shows the following:

- Both original dataset and outliers free dataset gives already high performance results on SVM, KNN and Random Forest 
- Outliers-free dataset outperformed the original dataset in all three classifiers
- Random forest and KNN shows sign of overfitting in the Outliers-free case
- Performance dropped significantly for the same classifiers on Collinearity-free dataset except for RBF (overfitting)
- PCA-reduced datasets gave very poor results
    -  PCA for MFCC not meaningful since MFCC is already a transformed data
    - In classification problem, when the differentiating characteristics of the classes are not reflected in variance of the variables, PCA may not be a good choice of data processing. This is because PCA does not take into account class information when calculating the principal components.
- LDA and Logistic regression are not optimal classifiers for the dataset (a sign of non-linear decision boundary between classes). Moreover:
  - as discussed before, predictors are coming from non gaussian distribution (another possible reason for the LDA to fail)
  - LR does not perform well in the presence of collinearity


```python
#learning curve plot (source sklearn)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, title1, estimator2, title2, estimator3, title3, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):

    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,6), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, scoring='f1_weighted', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="blue")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title(title1, fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, scoring='f1_weighted', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title(title2, fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, scoring='f1_weighted', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title(title3, fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    plt.tight_layout()
#     return plt
```


```python
estimator1 = SVC(gamma=0.01, C=10)

estimator2 = KNeighborsClassifier(3)

estimator3 = RandomForestClassifier(n_estimators=500, 
                                    max_features = 'auto', 
                                    max_depth = 40, 
                                    min_samples_split = 2, 
                                    min_samples_leaf = 1,
                                    bootstrap = False,
                                    criterion = 'entropy',
                                    random_state = 42)
```


```python
plot_learning_curve(estimator1, "Original dataset \nRBF learning Curve", 
                    estimator2, "Original dataset \nKNN Learning Curve", 
                    estimator3, "Original dataset \nRandomForest Learning Curve" , 
                    train_data, train_labels, (0.1, 1.01), cv=cv_, n_jobs=4)
```


    
![png](output_107_0.png)
    



```python
plot_learning_curve(estimator1, "Outliers-free \nRBF Learning Curve", 
                    estimator2, "Outliers-free \nKNN Learning Curve", 
                    estimator3, "Outliers-free \nRandomForest Learning Curve",
                    train_noisefree, trainlabels_noisefree, (0.1, 1.01), cv=cv_, n_jobs=4)
```


    
![png](output_108_0.png)
    


For both Original and Outliers-free splits

- RBF shows almost ideal learning case

- KNN shows bias(underfitting): In the beginning with a small sample size the model is supposed to achieve a very low training error. However, it fails to do so, i.e. it is not even able to learn/overfit the small training sample. 
Since increasing model complexity didn't help, KNN is underperformed

- Random forests : Training score is at its maximum regardless of training examples and Cross-validation score increases over time and gap between cross-validation score and training score indicates high variance scenario (sign of overfitting)

Random forests are built on decision trees, and decision trees are sensitive to class imbalance. Each tree is built on a bag, and each bag is a uniform random sample from the data (with replacement). Therefore each tree will be biased in the same direction and magnitude (on average) by class imbalance.

### Imbalanced dataset classification by modifying classifiers' cost function

By considering only original and Outlier-free datasets and best performed classifiers with best performed hyperparameters, the same classification implemented on modified classifiers' cost function by penalizing mistakes on the minority class using **class_weight='balanced'** in the SVC and RF while in KNN, **weight='distance'** weights points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.


```python
candidate_datasets = {
                'Original dataset':[train_data,train_labels,test_data,test_labels],
                'noise-free':[train_noisefree, trainlabels_noisefree, test_noisefree, testlabels_noisefree]
               }
```


```python
candidate_classifiers = {
        "Support Vector Classifier": SVC(),         #  (probability=False) is compatible with imbalanced dataset
        "KNearest": KNeighborsClassifier(),       #3 ok, 5 overfittting
                        }    
```


```python
candidate_params = {
        '1':svc_params,
        '2':knears_params,
             }
```


```python
#introducing cost function modifier parameters 
pen_svc_params = {'C': np.logspace(1, 1, 1), 
              'kernel': ['rbf'],
              'gamma':np.logspace(-2, -2, 1),
              'class_weight':['balanced'],
              'probability':[True]}

pen_knears_params = {"n_neighbors": list(range(3,7,2)), 
                 "weights":['distance']}

pen_params = {
        '1':pen_svc_params,
        '2':pen_knears_params,
             }
```


```python
d=imbalanced_clf(cv=cv_,ds=candidate_datasets,clf=candidate_classifiers,par=pen_params)
```


```python
column_names=[ 'Dataset split',
                'model',
                'run_time',
                'F1 CV score',
                'F1 score on test set',
                'Precision score on test set',
                'Recall score on test set',
                'best parameters'
             ]
```


```python
d.reindex(columns=column_names).sort_values(by='F1 score on test set')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset split</th>
      <th>model</th>
      <th>run_time</th>
      <th>F1 CV score</th>
      <th>F1 score on test set</th>
      <th>Precision score on test set</th>
      <th>Recall score on test set</th>
      <th>best parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Original dataset</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
      <td>0.80</td>
      <td>0.7992</td>
      <td>0.8058</td>
      <td>0.7980</td>
      <td>{'n_neighbors': 3, 'weights': 'distance'}</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Original dataset</td>
      <td>SVC</td>
      <td>0.01</td>
      <td>0.81</td>
      <td>0.8202</td>
      <td>0.8317</td>
      <td>0.8182</td>
      <td>{'C': 10.0, 'class_weight': 'balanced', 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise-free</td>
      <td>SVC</td>
      <td>0.01</td>
      <td>0.83</td>
      <td>0.8394</td>
      <td>0.8430</td>
      <td>0.8438</td>
      <td>{'C': 10.0, 'class_weight': 'balanced', 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise-free</td>
      <td>KNeighborsClassifier</td>
      <td>0.0</td>
      <td>0.79</td>
      <td>0.8732</td>
      <td>0.8751</td>
      <td>0.8750</td>
      <td>{'n_neighbors': 3, 'weights': 'distance'}</td>
    </tr>
  </tbody>
</table>
</div>




```python
estimator1 = SVC(gamma=0.01, C=10,class_weight='balanced',probability=True)

estimator2 = KNeighborsClassifier(n_neighbors=3)
estimator3 = KNeighborsClassifier(n_neighbors=3,weights='distance')

plot_learning_curve(estimator1, "RBF Penalized", estimator2, "KNN", estimator3, "KNN Penalized",
                    train_noisefree, trainlabels_noisefree, (0.1, 1.01), cv=cv_, n_jobs=4)
```


    
![png](output_119_0.png)
    


by modifying the **costs of misclassification** the performance increased for both dataset splits

**from the learning curve it is very clear how by modifying the cost function of KNN, the underfitting problem is solved**


# Dataset Balancing:

Since the size of the dataset is relatively small, only ***oversampling*** of the minority classes using two of the most common technique will be considered

## Synthetic Minority Oversampling Technique (SMOTE)

 Considering a sample $x_i$, a new sample $x_{new}$ will be generated considering its $k$ neareast-neighbors. Then, one of these nearest-neighbors $x_{zi}$ is selected and a sample is generated as follows:\[x_{new} = x_i + \lambda \times (x_{zi} - x_i)\]
 where $\lambda$ is a random number in the range [0, 1]. This interpolation will create a sample on the line between $x_{i}$ and $x_{zi}$
 
 <img src="images/9.1.png" alt="Drawing" style="width: 300px;" />

## Adaptive Synthetic (ADASYN) 

ADASYN uses the same algorithms to generate samples, the key difference is that ADASYN uses a **density distribution**, as a criterion to automatically decide the number of synthetic samples that must be generated for each minority sample by adaptively changing the weights of the different minority samples to compensate for the skewed distributions. SMOTE generates the same number of synthetic samples for each original minority sample.

For multiple classes classification, both ADASYN and SMOTE need information regarding the neighbourhood of each sample used for sample generation. They are using a **one-vs-rest** approach by selecting each targeted class and computing the necessary statistics against the rest of the data set which are grouped in a single class.

**imbalanced-learn** is a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance. It is compatible with scikit-learn.


```python
# conda install -c glemaitre imbalanced-learn
```


```python
# import imblearn
from imblearn.over_sampling import SMOTE 
X = df.drop('language_coded', axis=1)
Y = df['language_coded']
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
# df_balanced = pd.concat([pd.DataFrame(X_res), pd.DataFrame(Y_res, columns=['language_coded'])], axis=1) #data split#4
print('SMOTE over-sampling:')
print(Y_res.value_counts())
Y_res.value_counts().plot(kind='bar', title='Count (target)');
```

    SMOTE over-sampling:
    5    165
    4    165
    3    165
    2    165
    1    165
    0    165
    Name: language_coded, dtype: int64
    


    
![png](output_124_1.png)
    



```python
#plot 
from imblearn.over_sampling import ADASYN
from numpy import where

fig, axs = plt.subplots(3,2,figsize=(16,16))

# scatter plot of examples by class label
from numpy import array
g = array(predictors.values.tolist())
k = language.values.tolist() 
k=np.array(k).T

import itertools
colors = itertools.cycle(["tab:blue", "tab:brown", "tab:green", "tab:purple","tab:red","tab:orange"])
colors_ = itertools.cycle(["tab:brown", "tab:orange"])

from collections import Counter
counter=Counter(language)
for label, _ in counter.items():
    row_ix = where(k == label)[0]
    axs[0,0].scatter(g[row_ix, 0], g[row_ix, 9], label=str(label), color=next(colors)) #X1 ,X10 choosen from orange 'find informative projections'
    axs[0,0].set(xlabel='X1', ylabel='X10', title='Original')

for label in ['FR','US']:    
    row_ix = where(k == label)[0]
    axs[0,1].scatter(g[row_ix, 0], g[row_ix, 9], label=str(label), color=next(colors_))
    axs[0,1].set(xlabel='X1', ylabel='X10', title='US vs Original FR')

oversample = SMOTE()
g_smote, k_smote = oversample.fit_resample(g, k)

# for label, _ in counter.items():
for label in ['FR','US']:    
    row_ix = where(k_smote == label)[0]
    axs[1,1].scatter(g_smote[row_ix, 0], g_smote[row_ix, 9], label=str(label), color=next(colors_))
    axs[1,1].set(xlabel='X1', ylabel='X10', title='US vs SMOTE FR')
    
for label, _ in counter.items():
    row_ix = where(k_smote == label)[0]
    axs[1,0].scatter(g_smote[row_ix, 0], g_smote[row_ix, 9], label=str(label), color=next(colors)) #X1 ,X10 choosen from orange 'find informative projections'
    axs[1,0].set(xlabel='X1', ylabel='X10', title='SMOTE')

ada = ADASYN(random_state = 101)
g_adasyn, k_adasyn = ada.fit_resample(g, k)

for label in ['FR','US']:    
    row_ix = where(k_adasyn == label)[0]
    axs[2,1].scatter(g_adasyn[row_ix, 0], g_adasyn[row_ix, 9], label=str(label), color=next(colors_))
    axs[2,1].set(xlabel='X1', ylabel='X10', title='US vs ADASYN FR')
    
for label, _ in counter.items():
    row_ix = where(k_adasyn == label)[0]
    axs[2,0].scatter(g_adasyn[row_ix, 0], g_adasyn[row_ix, 9], label=str(label), color=next(colors)) #X1 ,X10 choosen from orange 'find informative projections'
    axs[2,0].set(xlabel='X1', ylabel='X10', title='ADA')
plt.show()
```


    
![png](output_125_0.png)
    


## Oversampled Dataset Classification

Cross-Validaion for oversampled dataset is tricky and need a bit of work on coding mainly **to avoid validating a training fold on synthetic data** as shown on the example below
**synthetic data should only be used for training**, and neiher for validation nor for testing 


<img src="images/5.1.png" alt="Drawing" style="width: 700px;" />


```python
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
```


```python
cv_ = StratifiedKFold(n_splits=splits, random_state=42, shuffle=True) 
```


```python
def bestclfparams_oversamp (mode,cv=cv_,datasets=candidate_datasets,clf=candidate_classifiers,par=candidate_params):
    
    df_results = pd.DataFrame()

    for key in datasets:
        train_pred=datasets[key][0]
        train_y=datasets[key][1]
        test_pred=datasets[key][2]
        test_y=datasets[key][3]

        X_train = train_pred.values
        X_test = test_pred.values
        y_train = train_y.values
        y_test = test_y.values
        
        accuracy_lst = []
        precision_lst = []
        recall_lst = []
        f1_lst = []
        auc_lst = []

#         for key,classifier,paramsname,params in zip(classifiers.items(),params.items()):
        for (key1,classifier),(key2,parameter) in zip(clf.items(),par.items()):
            # grid=GridSearchCV(classifier, parameter)
            start_time = time.time()

            grid=GridSearchCV(classifier, parameter,cv=cv,scoring='f1_macro')

            for train, test in cv.split(X_train, y_train):
                if mode=='sm':
                    pipeline = imbalanced_make_pipeline(SMOTE(random_state = 42,sampling_strategy='minority'), grid)
                if mode=='ada': 
                    pipeline = imbalanced_make_pipeline(ADASYN(random_state = 42), grid)
                    # pipeline = Pipeline([('sm', SMOTE(sampling_strategy='minority')), ('clf', grid)])
                model = pipeline.fit(X_train[train], y_train[train])
                best_est = grid.best_estimator_
                prediction = best_est.predict(X_train[test])
                # print(best_est)
                f1_lst.append(f1_score(y_train[test], prediction,average='weighted')) #weighted is the best calculated on KNN+6folds

            row = {
                'Dataset split': key,
                'model':classifier.__class__.__name__,
                'run_time': format(round((time.time() - start_time)/60,2)),
                'F1 CV score': round(np.mean(f1_lst), 3) * 100,
                'F1 score on test set': round(f1_score(y_test, best_est.predict(X_test), average='weighted'),4)*100,
                'Acc score on test set': round(accuracy_score(y_test, best_est.predict(X_test)),4)*100,
                'pr score on test set': round(precision_score(y_test, best_est.predict(X_test), average='weighted'),4)*100,
                'rec score on test set': round(recall_score(y_test, best_est.predict(X_test), average='weighted'),4)*100,
                'best parameters':grid.best_params_,
                   }
            df_results = df_results.append(row, ignore_index=True) 
    return df_results
```


```python
column_names=[ 'Dataset split',
                'model',
                'run_time',
                'F1 CV score',
                'F1 score on test set',
                'Acc score on test set',
                'pr score on test set',
                'rec score on test set',
                'best parameters'
             ]
```


```python
k5=bestclfparams_oversamp (mode='sm')
```


```python
k5.reindex(columns=column_names).sort_values(by='F1 score on test set')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset split</th>
      <th>model</th>
      <th>run_time</th>
      <th>F1 CV score</th>
      <th>F1 score on test set</th>
      <th>Acc score on test set</th>
      <th>pr score on test set</th>
      <th>rec score on test set</th>
      <th>best parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Original dataset</td>
      <td>KNeighborsClassifier</td>
      <td>0.01</td>
      <td>79.9</td>
      <td>79.11</td>
      <td>78.79</td>
      <td>81.02</td>
      <td>78.79</td>
      <td>{'n_neighbors': 3}</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Original dataset</td>
      <td>SVC</td>
      <td>15.2</td>
      <td>83.2</td>
      <td>81.32</td>
      <td>81.82</td>
      <td>82.07</td>
      <td>81.82</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise-free</td>
      <td>KNeighborsClassifier</td>
      <td>0.01</td>
      <td>75.3</td>
      <td>85.43</td>
      <td>85.42</td>
      <td>86.71</td>
      <td>85.42</td>
      <td>{'n_neighbors': 3}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise-free</td>
      <td>SVC</td>
      <td>13.88</td>
      <td>80.0</td>
      <td>86.18</td>
      <td>86.46</td>
      <td>86.32</td>
      <td>86.46</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
    </tr>
  </tbody>
</table>
</div>




```python
l=bestclfparams_oversamp (mode='ada')
```


```python
l.reindex(columns=column_names).sort_values(by='F1 score on test set')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset split</th>
      <th>model</th>
      <th>run_time</th>
      <th>F1 CV score</th>
      <th>F1 score on test set</th>
      <th>Acc score on test set</th>
      <th>pr score on test set</th>
      <th>rec score on test set</th>
      <th>best parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Original dataset</td>
      <td>KNeighborsClassifier</td>
      <td>0.02</td>
      <td>79.3</td>
      <td>76.84</td>
      <td>76.77</td>
      <td>83.02</td>
      <td>76.77</td>
      <td>{'n_neighbors': 3}</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Original dataset</td>
      <td>SVC</td>
      <td>0.22</td>
      <td>83.2</td>
      <td>81.07</td>
      <td>80.81</td>
      <td>82.98</td>
      <td>80.81</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise-free</td>
      <td>KNeighborsClassifier</td>
      <td>0.02</td>
      <td>76.5</td>
      <td>81.75</td>
      <td>81.25</td>
      <td>84.98</td>
      <td>81.25</td>
      <td>{'n_neighbors': 3}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise-free</td>
      <td>SVC</td>
      <td>0.18</td>
      <td>80.1</td>
      <td>83.88</td>
      <td>84.38</td>
      <td>83.62</td>
      <td>84.38</td>
      <td>{'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}</td>
    </tr>
  </tbody>
</table>
</div>



- SMOTE increased the performance of the models on the Outliers-free dataset only
- ADASYN increased only SVC for the outliers-free split 

   | Dataset split | Model | Method | F1-score |
   | --- | --- | --- | ---|
        | Outlier-free | KNN | SMOTE      | 0.854 |
                   | Outlier-free | RBF | SMOTE      | 0.862 |
                   | Outlier-free |KNN|Penalized | 0.873 |


Above is the top three performed models

# Conclusion

A sample of 329 signals, containing 165 US voice and 164 non-US voice(45 UK, 29 ES, and 30 for IT,GR and FR each), is analyzed for the purpose of accent recognition. For each signal, the mean vector of MFCC matrix is used as an input vector for pattern recognition. Then different classifiers are compared based on the MFCC feature. By comparison, Support Vector Machine (RBF) and K-nearest Kneighbour yield the highest average test F1-Score, after using 10-folds cross-validation.

by minimizing the influence of the majority class by giving the closest neighbours more weight, KNN leverages the score about 5%. 
By performing SMOTE and ADASYN on the minority classes during cross-validation, we managed to elevate both KNN and RBF F1-score abount 4% 

Cleaning the minimum outlier/extreme data proved to help increasing classifiers performance regardless whether these instances are actually noise or not  


```python

```
