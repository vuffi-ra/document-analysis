import math
import string
from operator import itemgetter

import numpy as np
from collections import defaultdict
from corpus import CorpusLoader
from nltk.stem.porter import PorterStemmer  # IGNORE:import-error


class AbsoluteTermFrequencies(object):
    """Klasse, die zur Durchfuehrung absoluter Gewichtung von Bag-of-Words
    Matrizen (Arrays) verwendet werden kann. Da Bag-of-Words zunaechst immer in
    absoluten Frequenzen gegeben sein muessen, ist die Implementierung dieser 
    Klasse trivial. Sie wird fuer die softwaretechnisch eleganten Unterstuetzung
    verschiedner Gewichtungsschemata benoetigt (-> Duck-Typing).   
    """

    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        # Gibt das NumPy Array unveraendert zurueck, da die Bag-of-Words Frequenzen
        # bereits absolut sind.
        return bow_mat

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'absolute'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)


class RelativeTermFrequencies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative Frequenzen.
    """

    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die relative Gewichtung einer Bag-of-Words Matrix (relativ im 
        Bezug auf Dokumente) durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        return bow_mat[:] / (bow_mat.sum(axis=1))[:, None]

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'relative'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)


class RelativeInverseDocumentWordFrequecies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative - inverse Dokument Frequenzen.
    """

    def __init__(self, vocabulary, category_wordlists_dict):
        """Initialisiert die Gewichtungsberechnung, indem die inversen Dokument
        Frequenzen aus dem Dokument Korpous bestimmt werden.
        
        Params:
            vocabulary: Python Liste von Woertern (das Vokabular fuer die 
                Bag-of-Words).
            category_wordlists_dict: Python dictionary, das zu jeder Klasse (category)
                eine Liste von Listen mit Woertern je Dokument enthaelt.
                Siehe Beschreibung des Parameters cat_word_dict in der Methode
                BagOfWords.category_bow_dict.
        """

        self.total_documents = sum(map(len, category_wordlists_dict.values()))
        documents_dict = defaultdict(int)

        for documents in category_wordlists_dict.values():
            for document in documents:
                words = set(document)
                for word in words:
                    documents_dict[word] += 1

        self.documents_containing = [documents_dict[w] for w in vocabulary]


    def weighting(self, bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        relative = bow_mat[:] / (bow_mat.sum(axis=1))[:, None]
        # np.full(bow_mat.shape, self.total_documents) with self.total_documents = 5, bow_mat.shape = (3, 3)
        #
        # | 5 5 5 |
        # | 5 5 5 |
        # | 5 5 5 |

        log_matrix = np.full(bow_mat.shape, self.total_documents) / np.array(self.documents_containing)[None, :]
        return relative * np.log(log_matrix)


    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'tf-idf'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)

def count_words(word_list):
    occurences = defaultdict(int)
    for x in word_list:
        occurences[x] += 1
    return occurences


class BagOfWords(object):
    """Berechnung von Bag-of-Words Repraesentationen aus Wortlisten bei 
    gegebenem Vokabular.
    """

    def __init__(self, vocabulary, term_weighting=AbsoluteTermFrequencies()):
        """Initialisiert die Bag-of-Words Berechnung
        
        Params:
            vocabulary: Python Liste von Woertern / Termen (das Bag-of-Words Vokabular).
                Die Reihenfolge der Woerter / Terme im Vokabular gibt die Reihenfolge
                der Terme in den Bag-of-Words Repraesentationen vor.
            term_weighting: Objekt, das die weighting(bow_mat) Methode implemeniert.
                Optional, verwendet absolute Gewichtung als Default.
        """
        self.__vocabulary = vocabulary
        self.__term_weighting = term_weighting

    def category_bow_dict(self, cat_word_dict):
        """Erzeugt ein dictionary, welches fuer jede Klasse (category)
        ein NumPy Array mit Bag-of-Words Repraesentationen enthaelt.
        
        Params:
            cat_word_dict: Dictionary, welches fuer jede Klasse (category)
                eine Liste (Dokumente) von Listen (Woerter) enthaelt.
                cat : [ [word1, word2, ...],  <--  doc1
                        [word1, word2, ...],  <--  doc2
                        ...                         ...
                        ]
        Returns:
            category_bow_mat: Ein dictionary mit Bag-of-Words Matrizen fuer jede
                Kategory. Eine Matrix enthaelt in jeder Zeile die Bag-of-Words 
                Repraesentation eines Dokuments der Kategorie. (d x t) bei d 
                Dokumenten und einer Vokabulargroesse t (Anzahl Terme). Die
                Reihenfolge der Terme ist durch die Reihenfolge der Worter / Terme
                im Vokabular (siehe __init__) vorgegeben.
        """
        result = {}
        for category, documents in cat_word_dict.items():
            rows = []
            for document in documents:
                occurences = count_words(document)
                rows.append([occurences[w] for w in self.__vocabulary])
            result[category] = self.__term_weighting.weighting(np.array(rows))
        return result


    @staticmethod
    def most_freq_words(word_list, n_words=None):
        """Bestimmt die (n-)haeufigsten Woerter in einer Liste von Woertern.
        
        Params:
            word_list: Liste von Woertern
            n_words: (Optional) Anzahl von haeufigsten Woertern (top n). Falls
                n_words mit None belegt ist, sollen alle vorkommenden Woerter
                betrachtet werden.
            
        Returns:
            words_topn: Python Liste, die (top-n) am haeufigsten vorkommenden 
                Woerter enthaelt. Die Sortierung der Liste ist nach Haeufigkeit
                absteigend.
        """
        occurences = count_words(word_list)
        return sorted(occurences.keys(), key=occurences.get, reverse=True)[:n_words]



class WordListNormalizer(object):

    def __init__(self, stoplist=None, stemmer=None):
        """Initialisiert die Filter
        
        Params: 
            stoplist: Python Liste von Woertern, die entfernt werden sollen
                (stopwords). Optional, verwendet NLTK stopwords falls None
            stemmer: Objekt, das die stem(word) Funktion implementiert. Optional,
                verwendet den Porter Stemmer falls None.
        """

        if stoplist is None:
            stoplist = CorpusLoader.stopwords_corpus()
        self.__stoplist = stoplist

        if stemmer is None:
            stemmer = PorterStemmer()
        self.__stemmer = stemmer
        self.__punctuation = string.punctuation
        self.__delimiters = ["''", '``', '--']


    def normalize_words(self, word_list):
        """Normalisiert die gegebenen Woerter nach in der Methode angwendeten
        Filter-Regeln (Gross-/Kleinschreibung, stopwords, Satzzeichen, 
        Bindestriche und Anfuehrungszeichen, Stemming)
        
        Params: 
            word_list: Python Liste von Worten.
            
        Returns:
            word_list_filtered, word_list_stemmed: Tuple von Listen
                Bei der ersten Liste wurden alle Filterregeln, bis auch stemming
                angewandt. Bei der zweiten Liste wurde zusaetzlich auch stemming
                angewandt.
        """
        filter_set = set(self.__stoplist + list(self.__punctuation) + self.__delimiters)

        lowered = [s.lower() for s in word_list]
        word_list_filtered = [s for s in lowered if s not in filter_set]
        word_list_stemmed = [self.__stemmer.stem(s) for s in word_list_filtered]

        return word_list_filtered, word_list_stemmed


class IdentityFeatureTransform(object):
    """Realisert eine Transformation auf die Identitaet, bei der alle Daten 
    auf sich selbst abgebildet werden. Die Klasse ist hilfreich fuer eine
    softwaretechnisch elegante Realisierung der Funktionalitaet "keine Transformation
    der Daten durchfuehren" (--> Duck-Typing).
    """

    def estimate(self, train_data, train_labels):
        pass

    def transform(self, data):
        return data


class TopicFeatureTransform(object):
    """Realsiert statistische Schaetzung eines Topic Raums und Transformation
    in diesen Topic Raum.
    """

    def __init__(self, topic_dim):
        """Initialisiert die Berechnung des Topic Raums
        
        Params:
            topic_dim: Groesse des Topic Raums, d.h. Anzahl der Dimensionen.
        """
        self.__topic_dim = topic_dim
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None

    def estimate(self, train_data, train_labels):  # IGNORE:unused-argument
        """Statistische Schaetzung des Topic Raums
        
        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
                Hinweis: Fuer den hier zu implementierenden Topic Raum werden die
                Klassenlabels nicht benoetigt. Sind sind Teil der Methodensignatur
                im Sinne einer konsitenten und vollstaendigen Verwaltung der zur
                Verfuegung stehenden Information.
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')

    def transform(self, data):
        """Transformiert Daten in den Topic Raum.
        
        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x topic_dim).
        """
        raise NotImplementedError('Implement me')
