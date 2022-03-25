# Speaker Accent Recognition Using MFCCs
## Background
Accent recognition is classification of the speaker accent from an input signal. Classifying accents can provide information about a speaker’s nationality and heritage, which can help identify topics more relevant to the user, for the purposes of search results and advertisements. Typical applications include online banking, telephone shopping, and security applications.
Typically, the input signal is represented in frequency domain then dimensionality reduction can be performed together with feature extraction.

   ![1](https://user-images.githubusercontent.com/59888707/159986382-517ef236-3db3-4b84-b551-c22efe851805.jpg)

A common feature extraction technique for the purpose of speech recognition is Mel-frequency cepstral coefficients or MFCCs. MFCCs are coefficients that collectively make up an MFC (mel-frequency cepstrum) which is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
The main idea of MFCC is to transform the signal from time domain to frequency domain and to map the transformed signal in hertz onto Mel-scale due to the fact that 1 kHz is a threshold of humans’ hearing ability.

MFCCs are commonly derived as follows: 

1.	Take the absolute value of the short time Fourier transform of (a windowed excerpt of) a signal.
2.	Map the powers of the spectrum obtained above onto the mel scale
3.	Take the logs of the powers at each of the mel frequencies.
4.	Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
5.	The MFCCs are the amplitudes of the resulting spectrum.
6.	Return the first q MFCCs.

MFCC values are not very robust in the presence of additive noise, and so it is common to normalise their values in speech recognition systems to lessen the influence of noise. (energy terms)
![2](https://user-images.githubusercontent.com/59888707/159986475-56cfc591-71c9-4525-8715-f2de0247d717.png)

## 2. Dataset
   ### 2.1 Source
   
A total of 329 signal data were collected from the voice of 22 speakers 11 female and 11 male of accented speakers speaking English, containing 165 US voice and 164 non-US voice from 5 countries: Spain, France, Germany, Italy, and UK.

The sound tracks have lengths of only around 1 second, with a sampling rate of 44,100 Hz, each sound track vector on the time domain has more than 30,000 entries. Because of the method used in collecting the data, there is no background noise in any sound tracks.

The 12th lowest order melfrequency cepstral coefficients (MFCCs) of the audio signals are used as inputs to the algorithms.

The Source of both audio files and MFCC spreadsheet available at UCI at the following link https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition

   ### 2.2 Problem formulation
   
This accent reognition is a classification problem and the response variable $yi$ is givn by:

![tex2img_equation](https://user-images.githubusercontent.com/59888707/160080894-276fb852-00b2-4c5c-9aa7-4ef5c3ab73b4.png)

showing that there are 6 class labels 
The design is **balanced in terms of US/NOT US accent but we want to extend the problem to classify all accents, hence imbalanced problem**

## 3. Tools
language: Python 3.8.5

platform: Jupyter

Libraries: (numpy, pandas, Matplotlib, seaborn, scipy, sklearn, imblearn)
      
## 4. Results

Code available on
