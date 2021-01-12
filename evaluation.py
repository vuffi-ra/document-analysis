import numpy as np
from features import IdentityFeatureTransform
from collections import defaultdict

class CrossValidation(object):
    
    def __init__(self, category_bow_dict, n_folds):
        """Initialisiert die Kreuzvalidierung ueber gegebnen Daten
        
        Params:
            category_bow_dict: Dictionary, das fuer jede Klasse ein ndarray mit Merkmalsvektoren
                (zeilenweise) enthaelt.
            n_folds: Anzahl von Ausschnitten ueber die die Kreuzvalidierung berechnet werden soll.

        """
        self.__category_bow_list = list(category_bow_dict.items())
        self.__n_folds = n_folds
        
    def validate(self, classifier, feature_transform=None):
        """Berechnet eine Kreuzvalidierung ueber die Daten,
        
        Params:
            classifier: Objekt, das die Funktionen estimate und classify implementieren muss.
            feature_transform: Objekt, das die Funktionen estimate und transform implementieren 
                muss. Optional: Falls None, wird keine Transformation durchgefuehrt.

        Returns:
            crossval_overall_result: Liste von [0, 1] Bewertungen über alle Kategorien über alle Testläufe
            crossval_class_results: Liste von (category, Liste von [0, 1] Bewertungen über alle Testläufe)
        """
        if feature_transform is None:
            feature_transform = IdentityFeatureTransform()

        # Shape
        # [
        #   [ error_rate, n_wrong, n_samples],
        #   [ error_rate, n_wrong, n_samples],
        #   [ error_rate, n_wrong, n_samples],
        #   [ error_rate, n_wrong, n_samples],
        #   [ error_rate, n_wrong, n_samples],
        # ]
        crossval_overall_list = []
        # Shape
        # {
        #   "adventure": [
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #   ],
        #   "news": [
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #      [error_rate, n_wrong, n_samples]
        #   ],
        # }

        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.__n_folds):
            train_bow, train_labels, test_bow, test_labels = self.corpus_fold(fold_index)

            feature_transform.estimate(train_bow, train_labels)

            train_feat = feature_transform.transform(train_bow)
            test_feat = feature_transform.transform(test_bow)

            classifier.estimate(train_feat, train_labels)
            estimated_test_labels = classifier.classify(test_feat)

            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            crossval_class_list = classifier_eval.category_error_rates()

            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results
        
    @staticmethod
    def __crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result
        
        
        
    def corpus_fold(self, fold_index):
        """Berechnet eine Aufteilung der Daten in Training und Test
        
        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.
        
        Returns:
            training_bow_mat: BoW for documents not in test data
            training_label_mat: Categories for documents not in test data
            test_bow_mat: BoW for 1/__n_folds of documents behind fold_index
            test_label_mat: Categories for 1/__n_folds of behind fold_index
        """
        training_bow_mat = []
        training_label_mat = []
        test_bow_mat = []
        test_label_mat = []
        
        for category, bow_mat in self.__category_bow_list:
            n_category_samples = bow_mat.shape[0]
            #
            # Erklaeren Sie nach welchem Schema die Aufteilung der Daten erfolgt.
            #
            # Select indices for fold_index-th test fold, remaining indices are used for training
            test_indices = list(range(fold_index, n_category_samples, self.__n_folds))
            train_indices = [train_index for train_index in range(n_category_samples) 
                             if train_index not in test_indices]
            category_train_bow = bow_mat[train_indices, :]
            category_test_bow = bow_mat[test_indices, :]
            # Construct label matrices ([x]*3 --> [x, x, x])
            category_train_labels = np.array([[category] * len(train_indices)])
            category_test_labels = np.array([[category] * len(test_indices)])

            training_bow_mat.append(category_train_bow)
            training_label_mat.append(category_train_labels.T)
            test_bow_mat.append(category_test_bow)
            test_label_mat.append(category_test_labels.T)

        training_bow_mat = np.vstack(tuple(training_bow_mat))
        training_label_mat = np.vstack(tuple(training_label_mat))
        test_bow_mat = np.vstack(tuple(test_bow_mat))
        test_label_mat = np.vstack(tuple(test_label_mat))

        return training_bow_mat, training_label_mat, test_bow_mat, test_label_mat



class ClassificationEvaluator(object):
    
    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis 
        auf Testdaten.
        
        Params:
            estimated_labels: ndarray (N x 1) mit durch den Klassifikator 
                bestimmten Labels.
            groundtruth_labels: ndarray (N x 1) mit den tatsaechlichen Labels.
                
        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        #
        # Bestimmen Sie hier die Uebereinstimmungen und Abweichungen der
        # durch den Klassifikator bestimmten Labels und der tatsaechlichen 
        # Labels
        
        self.comparison = self.__groundtruth_labels == self.__estimated_labels


    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.
        
        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        if mask is None:
            mask = np.array([True] * len(self.__groundtruth_labels)).reshape(-1, 1)

        n_samples = np.sum(mask)
        n_wrong = np.sum(~self.comparison[mask])

        return (n_wrong / n_samples) * 100.0, n_wrong, n_samples



    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten
        
        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        result = []
        for category in set(self.__groundtruth_labels.reshape(-1).tolist()):
            mask = self.__groundtruth_labels == np.array([category] * len(self.__groundtruth_labels)).reshape(-1, 1)
            error_rate, n_wrong, n_samples = self.error_rate(mask)
            result.append((category, error_rate, n_wrong, n_samples))
        return sorted(result, key=lambda x: x[0])



