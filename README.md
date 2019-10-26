VUA Explore
===========
*Jenia Kim*

This repo contains the VuaExploration class used for data exploration and
descriptive statistics of textual data in the [VUA format](#VUA-format)
(used by the [CLTL](http://www.cltl.nl/) at VU Amsterdam).

See the [example.ipynb](https://github.com/vanboefer/vua_explore/blob/master/example.ipynb) notebook for usage demonstration.

The following data analyses are included:
- text length
- word counts and most frequent words per label
- association measures between words and labels (PMI)
- proportions of overlap with a lexicon
- t-SNE plots

## Data
The repo does not contain the data used in [example.ipynb](https://github.com/vanboefer/vua_explore/blob/master/example.ipynb).
The data is available online (but needs to be converted to VUA format):
- [davidson](https://github.com/t-davidson/hate-speech-and-offensive-language)
- [gibert](https://github.com/aitor-garcia-p/hate-speech-dataset)
- [qian](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech)
- [wulczyn](https://figshare.com/articles/Wikipedia_Detox_Data/4054689)

## VUA format
- csv file (sep='\t')
- columns: 'Id', 'Text', 'Label'
