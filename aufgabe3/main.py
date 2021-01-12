import itertools
import math
from collections import defaultdict

import numpy as np

from classification import KNNClassifier
from corpus import CorpusLoader
from evaluation import CrossValidation
from features import BagOfWords, WordListNormalizer, RelativeTermFrequencies, RelativeInverseDocumentWordFrequecies, \
    AbsoluteTermFrequencies
from visualization import hbar_plot



def aufgabe3():
    # ********************************** ACHTUNG **************************************
    # Die nun zu implementierenden Funktionen spielen eine zentrale Rolle im weiteren 
    # Verlauf des Fachprojekts. Achten Sie auf eine effiziente und 'saubere' Umsetzung. 
    # Verwenden Sie geeignete Datenstrukturen und passende Python Funktionen.
    # Wenn Ihnen Ihr Ansatz sehr aufwaendig vorkommt, haben Sie vermutlich nicht die
    # passenden Datenstrukturen / Algorithmen / (highlevel) Python / NumPy Funktionen
    # verwendet. Fragen Sie in diesem Fall!
    #
    # Schauen Sie sich jetzt schon gruendlich die Klassen und deren Interfaces in den
    # mitgelieferten Modulen an. Wenn Sie Ihre Datenstrukturen von Anfang an dazu 
    # passend waehlen, erleichtert dies deren spaetere Benutzung. Zusaetzlich bieten 
    # diese Klassen bereits etwas Inspiration fuer Python-typisches Design, wie zum 
    # Beispiel Duck-Typing.
    #
    # Zu einigen der vorgebenen Intefaces finden Sie Unit Tests in dem Paket 'test'. 
    # Diese sind sehr hilfreich um zu ueberpruefen, ob ihre Implementierung zusammen
    # mit anderen mitgelieferten Implementierungen / Interfaces funktionieren wird.
    # Stellen Sie immer sicher, dass die Unit tests fuer die von Ihnen verwendeten 
    # Funktionen erfolgreich sind. 
    # Hinweis: Im Verlauf des Fachprojekts werden die Unit Tests nach und nach erfolg-
    # reich sein. Falls es sie zu Beginn stoert, wenn einzelne Unit Tests fehlschlagen
    # koennen Sie diese durch einen 'decorator' vor der Methodendefinition voruebergehend
    # abschalten: @unittest.skip('')
    # https://docs.python.org/3/library/unittest.html#skipping-tests-and-expected-failures
    # Denken Sie aber daran sie spaeter wieder zu aktivieren.
    #
    # Wenn etwas unklar ist, fragen Sie!     
    # *********************************************************************************

    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()
    brown_categories = brown.categories()

    # Um eine willkuerliche Aufteilung der Daten in Training und Test zu vermeiden,
    # (machen Sie sich bewusst warum das problematisch ist)
    # verwendet man zur Evaluierung von Klassifikatoren eine Kreuzvalidierung.
    # Dabei wird der gesamte Datensatz in k disjunkte Ausschnitte (Folds) aufgeteilt.
    # Jeder dieser Ausschnitte wird einmal als Test Datensatz verwendet, waehrend alle
    # anderen k-1 Ausschnitte als Trainingsdatensatz verwendet werden. Man erhaehlt also
    # k Gesamtfehlerraten und k klassenspezifische Fehlerraten die man jeweils zu einer
    # gemeinsamen Fehlerrate fuer die gesamte Kreuzvalidierung mittelt. Beachten Sie, 
    # dass dabei ein gewichtetes Mittel gebildet werden muss, da die einzelnen Test Folds
    # nicht unbedingt gleich gross sein muessen.

    # Fuehren Sie aufbauend auf den Ergebnissen aus aufgabe2 eine 5-Fold Kreuzvalidierung 
    # fuer den k-Naechste-Nachbarn Klassifikator auf dem Brown Corpus durch. Dazu koennen 
    # Sie die Klasse CrossValidation im evaluation Modul verwenden. 
    #
    # Vollziehen Sie dazu nach wie die Klasse die Daten in Trainging und Test Folds aufteilt.
    # Fertigen Sie zu dem Schema eine Skizze an. Diskutieren Sie Vorteile und Nachteile.
    # Schauen Sie sich an, wie die eigentliche Kreuzvalidierung funktioniert. Erklaeren Sie
    # wie das Prinzip des Duck-Typing hier angewendet wird.
    #
    # Hinweise: 
    #
    # Die Klasse CrossValidator verwendet die Klasse ClassificationEvaluator, die Sie schon
    # fuer aufgabe2 implementieren sollten. Kontrollieren Sie Ihre Umsetzung im Sinne der
    # Verwendung im CrossValidator.
    #
    # Fuer das Verstaendnis der Implementierung der Klasse CrossValidator ist der Eclipse-
    # Debugger sehr hilfreich.

    words = [word for c in brown_categories for f in brown.fileids(c) for word in brown.words(f)]
    _, stemmed = WordListNormalizer().normalize_words(words)
    most_freq = BagOfWords.most_freq_words(stemmed, 20)

    dict = {}
    for c in brown_categories:
        dict[c] = [[w for w in brown.words(f)] for f in brown.fileids(c)]
    category_bow_dict = BagOfWords(most_freq).category_bow_dict(dict)

    validation_result = CrossValidation(category_bow_dict, 5).validate(KNNClassifier(k_neighbors=1, metric="euclidean"))
    print(validation_result)

    # Bag-of-Words Weighting 
    #
    # Bisher enthalten die Bag-of-Words Histogramme absolute Frequenzen.
    # Dadurch sind die Repraesentationen abhaengig von der absoluten Anzahl
    # von Woertern in den Dokumenten.
    # Dies kann vermieden werden, indem man die Bag-of-Words Histogramme mit
    # einem Normalisierungsfaktor gewichtet. 
    # 
    # Normalisieren Sie die Bag-of-Words Histogramme so, dass relative Frequenzen
    # verwendet werden. Implementieren und verwenden Sie die Klasse RelativeTermFrequencies 
    # im features Modul. 
    #
    # Wie erklaeren Sie das Ergebnis? Schauen Sie sich dazu noch einmal die 
    # mittelere Anzahl von Woertern pro Dokument an (aufgabe2).
    #
    # Wie in der Literatur ueblich, verwenden wir den
    # Begriff des "Term". Ein Term bezeichnet ein Wort aus dem Vokabular ueber
    # dem die Bag-of-Words Histogramme gebildet werden. Ein Bag-of-Words Histogramm
    # wird daher auch als Term-Vektor bezeichnet.

    category_bow_dict = BagOfWords(most_freq, RelativeTermFrequencies()).category_bow_dict(dict)

    validation_result = CrossValidation(category_bow_dict, 5).validate(KNNClassifier(k_neighbors=1, metric="euclidean"))
    print(validation_result)
    # Average length across all categories: 2343
    # Category, Absolute Error, Relative Error, Average Length
    # Adventure,       65.51    65.51    2391.10
    # Belle Lettres,   69.33    66.66    2307.94
    # Editorial,       70.37    51.85    2281.62
    # Fiction,         68.96    62.06    2361.65
    # Government,      73.33    69.99    2337.23
    # Hobbies,         88.88    72.22    2287.36
    # Humor,           77.77    88.88    2410.55
    # Learned,         55.0,    73.75    2273.6
    # Lore,            83.33    81.25    2297.89
    # Mistery,         87.5,    79.16    2382.04
    # News,            38.63    54.54    2285.31
    # Religion,        88.23    82.35    2317.58
    # Reviews,         41.17    35.29    2394.35
    # Romance,         79.31    79.31    2414.55
    # Science Fiction, 99.99    99.99    2411.66

    # Dokumente werden nicht mehr über länge klassifiziert, sondern stärker nach Inhalt.
    # Deshalb werden durchschnittlich lange Kategorien nun besser klassifierziert.

    # Zusaetzlich kann man noch die inverse Frequenz von Dokumenten beruecksichtigen
    # in denen ein bestimmter Term vorkommt. Diese Normalisierung wird als  
    # inverse document frequency bezeichnet. Die Idee dahinter ist Woerter die in
    # vielen Dokumenten vorkommen weniger stark im Bag-of-Words Histogramm zu gewichten.
    # Die zugrundeliegende Annahme ist aehnlich wie bei den stopwords (aufgabe1), dass 
    # Woerter, die in vielen Dokumenten vorkommen, weniger Bedeutung fuer die 
    # Unterscheidung von Dokumenten in verschiedene Klassen / Kategorien haben als
    # Woerter, die nur in wenigen Dokumenten vorkommen. 
    # Diese Gewichtung laesst sich statistisch aus den Beispieldaten ermitteln.
    #
    # Zusammen mit der relativen Term Gewichtung ergibt sich die so genannte
    # "term frequency inverse document frequency"
    #
    #                            Anzahl von term in document                       Anzahl Dokumente
    # tfidf( term, document )  = ----------------------------   x   log ( ---------------------------------- ) 
    #                             Anzahl Woerter in document              Anzahl Dokumente die term enthalten
    #
    # http://www.tfidf.com
    #
    # Eklaeren Sie die Formel. Plotten Sie die inverse document frequency fuer jeden 
    # Term ueber dem Brown Corpus.
    #
    # Wort in jedem Dokument => log(1) = 0
    # Wort in nur einem Dokument => log(Sehr Groß) > 1
    #
    # Implementieren und verwenden Sie die Klasse RelativeInverseDocumentWordFrequecies
    # im features Modul, in der Sie ein tfidf Gewichtungsschema umsetzen.
    # Ermitteln Sie die Gesamt- und klassenspezifischen Fehlerraten mit der Kreuzvalidierung.
    # Vergleichen Sie das Ergebnis mit der absoluten und relativen Gewichtung.
    # Erklaeren Sie die Unterschiede in den klassenspezifischen Fehlerraten. Schauen Sie 
    # sich dazu die Verteilungen der Anzahl Woerter und Dokumente je Kategorie aus aufgabe1
    # an. In wie weit ist eine Interpretation moeglich?

    total_documents = sum(map(len, dict.values()))
    documents_dict = defaultdict(int)

    for documents in dict.values():
        for document in documents:
            doc_words = set(document)
            for word in doc_words:
                documents_dict[word] += 1

    values = [math.log(total_documents / documents_dict[w]) for w in most_freq if documents_dict[w] != 0]
    #hbar_plot(values, title="IDS")

    category_bow_dict = BagOfWords(most_freq, RelativeInverseDocumentWordFrequecies(most_freq, dict)).category_bow_dict(dict)

    validation_result = CrossValidation(category_bow_dict, 5).validate(KNNClassifier(k_neighbors=1, metric="euclidean"))
    print(validation_result)

    # Evaluieren Sie die beste Klassifikationsleistung   
    #
    # Ermitteln Sie nun die Parameter fuer die beste Klassifikationsleistung des
    # k-naechste-Nachbarn Klassifikators auf dem Brown Corpus mit der Kreuzvalidierung.
    # Dabei wird gleichzeitig immer nur ein Parameter veraendert. Man hat eine lokal
    # optimale Parameterkonfiguration gefunden, wenn jede Aenderung eines Parameters
    # zu einer Verschlechterung der Fehlerrate fuehrt.
    #
    # Erlaeutern Sie warum eine solche Parameterkonfiguration lokal optimal ist.
    # 
    # Testen Sie mindestens die angegebenen Werte fuer die folgenden Parameter:
    # 1. Groesse des Vokabulars typischer Woerter (100, 500, 1000, 2000)
    # 2. Gewichtung der Bag-of-Words Histogramme (absolute, relative, relative with inverse document frequency)
    # 3. Distanzfunktion fuer die Bestimmung der naechsten Nachbarn (Cityblock, Euclidean, Cosine)
    # 4. Anzahl der betrachteten naechsten Nachbarn (1, 2, 3, 4, 5, 6)
    #
    # Erklaeren Sie den Effekt aller Parameter. 
    #
    # Erklaeren Sie den Effekt zwischen Gewichtungsschema und Distanzfunktion.

    params = [(words, weighting, metric, k)
              for words in [100, 500, 1000, 2000]
              for weighting in ["absolute", "relative", "fids"]
              for metric in ["euclidean", "cityblock", "cosine"]
              for k in [1, 2, 3, 4, 5, 6]]

    minimum_error_rate = 100.0
    found_params = None
    for i, param in enumerate(params):
        vocabulary = BagOfWords.most_freq_words(stemmed, param[0])

        if param[1] == "absolute":
            weighting = AbsoluteTermFrequencies()
        elif param[1] == "relative":
            weighting = RelativeTermFrequencies()
        else:
            weighting = RelativeInverseDocumentWordFrequecies(vocabulary, dict)

        category_bow_dict = BagOfWords(vocabulary, weighting).category_bow_dict(dict)
        validation_result = CrossValidation(category_bow_dict, 5).validate(KNNClassifier(k_neighbors=param[3], metric=param[2]))
        if validation_result[0] < minimum_error_rate:
            minimum_error_rate = validation_result[0]
            found_params = param
        print("Testing params ", i, " of ", len(params), ". ", param, ", Error Rate: ", validation_result[0])

    print(found_params)
    print(minimum_error_rate)

# Result
# (2000, 'relative', 'cityblock', 6)
# 55.800000000000004


if __name__ == '__main__':
    aufgabe3()
