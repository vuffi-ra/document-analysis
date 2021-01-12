import numpy as np
import scipy.spatial.distance
from features import BagOfWords, IdentityFeatureTransform



class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern
        
        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string), siehe auch scipy Funktion cdist 
        """
        self.__k_neighbors = k_neighbors
        self.__metric = metric
        # Initialisierung der Membervariablen fuer Trainingsdaten als None. 
        self.__train_samples = None
        self.__train_labels = None

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.
        
        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden. 
        Diskutieren Sie, was den K-NN stattdessen definiert.
        
        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

            [
                [ "Hallo", "ich", "bin", "Markus"]
                [ "Es", "war", "einmal"]
            ]
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            [
                "news"
                "märchen"
            ]
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.__train_samples = train_samples
        self.__train_labels = train_labels

    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__train_samples is None or self.__train_labels is None:
            raise ValueError('Classifier has not been "estimated", yet!')

        distances = scipy.spatial.distance.cdist(test_samples, self.__train_samples, metric=self.__metric)
        indices = np.argsort(distances)[:, :self.__k_neighbors]
        train_labels = self.__train_labels[indices]
        length = test_samples.shape[0]
        result = []
        for i in range(0, length):
            result.append(BagOfWords.most_freq_words(train_labels[i, :].reshape(-1).tolist())[0])
        return np.array(result).reshape((length, 1))


class BayesClassifier(object):

    def __init__(self):
        """Initialisiert den Multinomial Bayes Klassifikator
        """
        # ndarray mit Klassen a-priori Wahrscheinlichkeiten
        self.__cat_apriori = None
        # ndarray mit Term-Kategorie Wahrscheinlichkeiten
        self.__term_cat_probs = None
        # ndarray mit allen bekannten Klassen (unabhaengig von Labels fuer Daten)
        self.__cat_labels = None

    def estimate(self, train_samples, train_labels):
        """Trainiert den Multinomial Bayes Klassfikator.
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__cat_apriori is None or self.__term_cat_probs is None or \
                self.__cat_labels is None:
            raise ValueError('BayesClassifier has not been estimated!')

        raise NotImplementedError('Implement me')


