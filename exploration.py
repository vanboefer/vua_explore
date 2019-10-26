# standard library
import math
import string
from collections import Counter

# third party
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer


class VuaExploration(object):
    """
    VuaExploration
    ==============
    Class for data exploration and descriptive statistics of textual data.
    See example.ipynb for usage demonstration.

    vua format
    ----------
    The data needs to be in "vua format":
    - csv file (sep='\t'),
    - columns: 'Id', 'Text', 'Label'.
    Use the load_vua method to instantiate class from file.

    attributes
    ----------
    data
    labels
    stoplist
    punctlist
    tokenizer
    wordcounts
    length_stats
    lexicon
    lexicon_overlap

    methods
    -------
    get_wordcounts
    get_topwords
    get_pmi
    get_top_pmi
    tsne_plot
    load_vua
    """

    # `stoplist` and `punclist` are used to filter out
    # stop words and punctuation from the word counts
    stoplist = (
        stopwords.words('english') +
        ['rt', "i'm", 'u', "ain't", "n't", 'would', 'us', "that's", 'also']
    )
    punctlist = (
        list(string.punctuation) +
        ['...', '“', '”', '..', '…', '’', '::', '___', '♪', '♫']
    )


    def __init__(self, data, tokenizer=None, lexicon=None):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = self.data.Label.unique()
        self.lexicon = lexicon
        self.get_wordcounts()
        self.get_pmi()


    @property
    def length_stats(self):
        "Descriptive statistics of text lengths."

        return self.data.comment_length.describe()


    def get_wordcounts(self):
        "Return dictionary of word counts per label."

        results = dict()
        for label in self.labels:
            counts = Counter()
            lists_of_tokens = self.data.query(f"Label == @label").tokens
            for tokenlist in lists_of_tokens:
                for token in tokenlist:
                    if (
                        token not in self.punctlist and
                        token not in self.stoplist
                    ):
                        counts[token] += 1
            results[label] = counts

        self.wordcounts = results
        return results


    def get_topwords(self, label, n=20):
        "Return `n` most frequent words for `label`."

        return self.wordcounts[label].most_common(n)


    def get_pmi(self, min_freq=20):
        """
        Return dictionary of Pointwise Mutual Information (PMI) per label.
        Words that appear less than `min_freq` within a label are excluded.
        --------------------------------------------------------------------
        PMI is a measure of association between a word and a label.
        Formula: pmi(x;y) = log(p(x,y)/p(x)*p(y))
        more info: https://en.wikipedia.org/wiki/Pointwise_mutual_information
        """

        all_words = Counter()
        for counts in self.wordcounts:
            all_words.update(self.wordcounts[counts])
        total_words = sum(all_words.values())

        results = dict()
        for label in self.labels:
            p_y = len(self.data.loc[self.data.Label == label]) / total_words

            pmis = dict()
            for word in self.wordcounts[label]:
                if self.wordcounts[label][word] < min_freq:
                    continue
                p_xy = self.wordcounts[label][word] / total_words
                p_x = all_words[word] / total_words
                pmis[word] = math.log(p_xy / (p_x * p_y))
            results[label] = pmis

        self.pmi = results
        return results


    def get_top_pmi(self, label, n=10):
        "Return `n` highest PMI's for `label`."

        pmis = self.pmi[label]
        return sorted(pmis, key=pmis.get, reverse=True)[:n+1]


    def _find_lex_match(self, lexicon, case_sensitive=False):
        """
        Private function for checking whether the comments in the 'Text' column
        contain at least one match with the lexicon.
        Adds boolean series to the data.
        """

        concat_lex = '|'.join([rf"\b{item}\b" for item in lexicon])
        lex_regex = rf"({concat_lex})"
        self.data['lex_match'] = self.data.Text.str.contains(
            lex_regex,
            regex=True,
            case=case_sensitive,
        )


    @property
    def lexicon(self):
        "Lexicon used for checking overlap."
        return self._lexicon


    @lexicon.setter
    def lexicon(self, lexicon):
        if lexicon is not None:
            self._find_lex_match(lexicon)
        self._lexicon = lexicon


    @property
    def lexicon_overlap(self):
        "Percentage of comments with at least one lexicon match per label."

        if not self.lexicon:
            return None

        piv = self.data.pivot_table(
            values='Id',
            index='Label',
            columns='lex_match',
            margins=True,
            margins_name='Total',
            aggfunc='count',
        )
        piv['%True'] = (piv[True] / piv['Total'] * 100).round(1)

        return piv


    def tsne_plot(self, outpath, sample_size=1000, tfidf=True):
        """
        Creates a png file at `outpath` with t-SNE visualization.
        `sample_size` determines the size of the random sample from each label.
        Uses TfidfVectorizer by default;
        if `tfidf` is set to False, CountVectorizer is used.
        -----------------------------------------------------------------------
        More info:
        https://www.scikit-yb.org/en/latest/api/text/tsne.html
        https://lvdmaaten.github.io/tsne/
        """

        if self.tokenizer is None:
            print('No tokenizer was loaded.')
            return None

        df = pd.DataFrame(columns=self.data.columns)
        for label in self.labels:
            samp_df = self.data \
                .query("Label == @label") \
                .sample(sample_size, random_state=19)
            df = df.append(samp_df, ignore_index=True)

        # vectorize
        if tfidf:
            vectorizer = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)
        else:
            vectorizer = CountVectorizer(tokenizer=self.tokenizer.tokenize)
        X = vectorizer.fit_transform(df.Text)
        y = df.Label

        # create the visualizer and draw the vectors
        tsne = TSNEVisualizer()
        tsne.fit(X, y)
        tsne.show(outpath=outpath)

        return None


    @classmethod
    def load_vua(
        cls,
        path,
        sep='\t',
        preserve_case=False,
        reduce_len=True,
        strip_handles=True,
    ):
        """
        Load VuaExploration object from csv file in vua format from `path`.
        - Tokenize 'Text' column.
        - Calculate length of 'Text' by row.
        """

        df = pd.read_csv(path, sep=sep, header=0)
        tokenizer = TweetTokenizer(
            preserve_case=preserve_case,
            reduce_len=reduce_len,
            strip_handles=strip_handles,
        )
        df['tokens'] = df.Text.apply(lambda x: tokenizer.tokenize(x))
        df['comment_length'] = df.tokens.apply(lambda x: len(x))
        return cls(df, tokenizer=tokenizer)
