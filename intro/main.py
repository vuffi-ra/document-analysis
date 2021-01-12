import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter


class RandomArrayGenerator(object):

    def __init__(self, seed=None):
        """ Initialisiert den Zufallsgenerator
        Params:
            seed: Zufallssaat
        """
        if seed is not None:
            np.random.seed(seed)

    def rand_uniform(self, arr_shape, min_elem=0, max_elem=1):
        """ Generiert eine Liste mit gleichverteilten Zufallszahlen
        Params:
            n_elem: Anzahl von Elementen in der Liste
            min_elem: Kleinstmoegliches Element 
            max_elem: Groesstmoegliches Element
        Returns: NumPy Array mit Zufallszahlen
            
        """
        rand_arr = np.random.rand(*arr_shape)
        rand_arr = min_elem + rand_arr * (max_elem - min_elem)
        return rand_arr

    def rand_gauss(self, arr_shape, mean=0, std_deviation=1):
        """ Generiert eine Liste mit normalverteilten Zufallszahlen
        Params:
            n_elem: Anzahl von Elementen in der Liste
            min_elem: Kleinstmoegliches Element 
            max_elem: Groesstmoegliches Element
        Returns: NumPy Array mit Zufallszahlen
            
        """
        # Der * Operator expandiert ein tuple Objekt in eine Parameter Liste
        # https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists
        rand_arr = np.random.randn(*arr_shape)
        rand_arr = mean + rand_arr * std_deviation
        return rand_arr


class PythonIntro(object):

    def datatypes(self, var_tup=None):
        print('\n[PythonIntro::datatypes]')

        #
        print('Elementare Datenstrukturen')
        #
        # Variablen sind dynamisch getypt

        # Int
        variable = 1
        print(type(variable))

        # Float
        variable = 1.0
        print(type(variable))

        # string
        variable = '1.0'
        print(type(variable))

        print('\nDatenstrukturen fuer Sequenzen')
        # tuple
        variable = (1, 1.0, '1.0')
        print(type(variable))

        if var_tup is None:
            var_tup = variable

        # Ueberpruefen Sie, ob sich Objekte mit den Werten 42 oder 23 in dem tuple oder
        # nicht in dem tuple befinden. Verwenden Sie dazu if-elif-else Abfragen mit den 
        # konditionalen Ausdruecken or, in / not in.
        # https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not
        # https://docs.python.org/3/library/stdtypes.html#sequence-types-str-unicode-list-tuple-bytearray-buffer-xrange
        # Geben Sie das Ergebnis mit print auf der Konsole aus.
        # Legen Sie zusaetzlich eine boolsche Variable mit dem entsprechenden Ergebnis an
        # und geben Sie diese am Ende dieser Methode als Ergebnis zurueck.

        result = False
        if (42 in var_tup) or (23 in var_tup):
            result = True

        # Wichtig: string und tuple sind nicht veraenderbar (immutable). tuples
        # eignen sich somit sehr gut als Container zum uebergeben / zurueckgeben 
        # von Werten in / aus Funktionen.
        #
        # Warum ist das ein Vorteil?

        try:
            variable[0] = 2
        except TypeError as err:
            print('Damit haben wir gerechnet...')
            print(err)

            # Im Gegensatz zu tuple ist der list Datentyp veraenderbar (mutable)
        # list
        variable = [1, 1.0, '1.0']
        print(type(variable))
        variable[0] = 2
        print(variable)

        # Geben Sie nun das Ergebnis, aus der if-Abfrage oben, als Ergebnis mit 
        # return zurueck
        return result

    def sequences(self, seq_start=3, seq_end=7, seq_step=1):
        print('\n[PythonIntro::sequences]')
        # Machen Sie sich mit Zugriffsmethoden fuer list / tuple Objekte vertraut
        # Dazu verwenden wir eine etwas laengere Liste. Mit der range Funktion kann 
        # man eine fortlaufende Liste von int Objekten erzeugen.
        # https://docs.python.org/3/library/functions.html#range
        test_list = list(range(seq_start, seq_end, seq_step))
        print(test_list)

        # Im Folgenden wird der Zugriff auf Listen geuebt. Fuer jede Aufgabe sollen
        # Sie das Ergenis in eine andere Variable schreiben, bzw. das Ergebnis
        # direkt zu einer Ergebnisliste hinzufuegen. 
        #
        # Bestimmen Sie

        #     die Anzahl von Elementen in der Liste test_list mit der Funktion len,
        #        https://docs.python.org/2/library/functions.html#len
        #     das erste Element, 
        #     das letze Element, 
        #     das vorletzte und letzte Element (Reihenfolge!), 
        #     das erste Drittel,
        #     das zweite Drittel,
        #     das letze Drittel. 
        #
        # https://docs.python.org/3/tutorial/introduction.html#lists
        # https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types
        # Auf das letzte Element greifen Sie mit -1 zu.
        #
        # Achten Sie auf den Datentyp, den sie bei der Berechnung der Indexposition 
        # verwenden. Verwenden Sie int(), float() fuer Typecasts.
        # https://docs.python.org/3/library/functions.html (siehe int() und float())
        #
        # Speichern Sie die Ergebnisse Ihrer Zugriffe jeweils als Elemente in einer 
        # Liste ab.
        length = len(test_list)
        result_list = [
            length,
            test_list[0],
            test_list[-1],
            test_list[-2:],
            test_list[:4],
            test_list[4:8],
            test_list[8:]
        ]

        # Wenn Sie mit einzelnen Elementen eines list / tuple Objekts arbeiten 
        # koennen Sie eine for Schleife verwenden.  
        # https://docs.python.org/3/tutorial/controlflow.html
        # Geben Sie die Elemente von test_list einzeln in einer for-Schleife aus.
        #
        # Wenn Sie zusaetzlich mit dem Index eines Elements arbeiten wollen verwenden
        # Sie enumerate().
        # https://docs.python.org/3/library/functions.html#enumerate
        #
        # Geben Sie in der Schleife zu jedem Element noch dessen index aus.
        # Erzeugen Sie dazu String Objekte in *EXAKT* der Form '%d: %d ' fuer 
        # den Index und das Listenelement. Schreiben Sie diese Strings in eine 
        # neue Liste. Konstruieren Sie die strings exakt so wie angegeben.
        #
        # Hinweis: Verwenden Sie GENAU den oben angegebenen Formartierungs-String
        # '%d: %d '. Achten Sie auf die Leerzeichen.

        result_list_two = ['%d: %d ' % x for x in enumerate(test_list)]

        # Geben Sie nun *beide* Ergebnislisten mit return zurueck. Verwenden Sie dabei
        # die Reihenfolge, in der Sie die Listen erstellt haben. 
        #
        # Eine Funktion kann mehrere Werte zurueckgeben, die dann implizit als
        # Tupel behandelt werden:
        # return var1, var2, ..., varN

        return result_list, result_list_two

    def sequences_complex(self, test_seq=(1, 1.0, '1.0')):
        print('\n[PythonIntro::sequences_complex]')
        #
        print("Komplexere Operationen auf Sequenzen")
        #
        # Kommen wir nun noch einemal zu dem initialen Beispiel zurueck (Methode datatypes()).
        print(test_seq)
        # Geben Sie nun den Inhalt des list Objekt in folgender Form in der Konsole aus:
        # [ <type: value>, ... ]
        # Also konkret fuer unser Beispiel
        # [ <int: 1>, <float: 1.0>, <string, 1.0> ]
        # Verwenden Sie dazu eine for Schleife und die Methoden type(obj).__name__, str().
        # https://docs.python.org/2/library/functions.html (Gehe zu type() sowie str() )
        #
        # Konstruieren Sie einen string, den Sie am Ende ausgeben. Zwei strings lassen
        # sich mit + konkatenieren. SEHR SCHLECHTER STIL (siehe unten).

        #
        # Leider ist das Konkatenieren von strings mit + wahnsinnig ineffizient. Da
        # strings nicht veraenderbar sind (immutable) muss jedesmal neuer Speicher
        # reserviert werden und der Inhalt des alten Speichers kopiert werden. 
        #
        # Stattdessen verfuegen strings ueber eine join() Funktion.
        # https://docs.python.org/3/library/string.html#string.join
        # Dieser uebergibt man die zu konkatenierenden strings als Sequenz (z.B. tuple oder list)
        # Da wir unsere Datenstruktur iterativ aufbauen wollen und tuple nach der 
        # Erzeugung nicht veraenderbar sind, verwenden wir eine Liste.
        # https://docs.python.org/3/tutorial/introduction.html#lists
        #
        # Ein einfache und effiziente Methode strings zu formatieren bietet der % 
        # Operator fuer strings.
        # https://docs.python.org/3/library/stdtypes.html#string-formatting-operations
        # Verwenden Sie dabei die Conversion Option 's'.
        # type_val_str = '<%s, %s>' % (obj_type.__name__, obj)
        # Auf diese Weise laesst sich der + Operator zur Konkatenation von strings 
        # komplett vermeiden.
        #
        # Konstruieren Sie nun einen string in *EXAKT* der Form 
        # [ <type, value>, <type, value>, ..., <type, value> ]
        # fuer das test_seq Objekt. Verwenden Sie dabei den % Operator sowie die join Methode.
        # Achten Sie dabei auch auf Leerzeichen und Kommata.

        # Geben Sie diesen string als Ergebnis der Methode zurueck.  

        inner = ", ".join(['<%s, %s>' % (type(obj).__name__, str(obj)) for obj in test_seq])
        return "[ %s ]" % inner

    def list_comprehension(self, test_seq=(1, 1.0, '1.0')):
        print('\n[PythonIntro::list_comprehension]')
        #
        # List Comprehensions
        #
        # Eine effiziente und sehr maechtige Methode zum Generieren von Listen sind
        # list comprehensions.
        # https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        #
        # Erstellen Sie gemaess des vorherigen Beispiels eine Liste mit Typenamen fuer
        # die Elemente in test_list. 
        #
        # Verwenden Sie dann die zip Methode, um eine Liste von tuple Ojekten der 
        # Form [(type_name,obj),...] zu erhalten.
        # https://docs.python.org/3/library/functions.html (gehe zu zip() )
        # 
        # Verwenden Sie diese Liste mit einer list comprehension, um wieder eine Liste
        # von strings der Form <obj_type_name, obj> zu erzeugen. Das Ergbnis koennen 
        # Sie dann wieder mit string.join() erzeugen und dann mit print ausgeben.

        # Ersetzen Sie nun die eckigen Klammern Ihrer list comprehension durch runde Klammern.
        # Dabei erzeugen Sie einen Generator. Ein Generator erzeugt die Sequenz von Daten nicht
        # sofort, sondern nur nach Anfrage. Dies kann bei der Erzeugung von sehr langen Sequenzen
        # vorteilhaft fuer die Nutzung des Hauptspeichers sein. In der Uebung werden wir bei
        # grossen Datenmengen eher auf die Verwendung von NumPy (siehe Klasse NumpyIntro) anstatt
        # auf list comprehensions oder Generatoren setzen. Fuer kuerzere Sequenzen, koennen sie
        # jedoch nuetzlich sein und bieten eine kompakte Syntax.
        # https://docs.python.org/3/glossary.html#term-generator-expression

        # Geben Sie das Generator Objekt als Ergebnis der Methode zurueck.

        return ('<%s, %s>' % (type(obj).__name__, str(obj)) for obj in test_seq)

    def dictionaries(self, rand_list=None):
        print('\n[PythonIntro::dictionaries]')
        #
        # Nun schauen wir uns den dict Datentyp an.  
        # https://docs.python.org/3/tutorial/datastructures.html#dictionaries
        # Dabei handelt es sich um Hashmaps, die von einem Schluessel Objekt auf ein
        # Wert Objekt abbilden.
        if rand_list is None:
            # Gegeben sei die folgende zufaellig erzeugte Liste.
            rand_arr_gen = RandomArrayGenerator()
            rand_arr = rand_arr_gen.rand_gauss(arr_shape=(40,), mean=20, std_deviation=5)
            rand_list = list(rand_arr)
        # Schauen Sie sich bei der Gelegenheit objektorientierte Konzepte in Python an:
        # siehe Definition der Klasse RandomArrayGenerator oben.
        # https://docs.python.org/3/tutorial/classes.html
        # Schauen Sie sich dabei auch Default-Argumente an.
        # https://docs.python.org/3/tutorial/controlflow.html#default-argument-values
        # Bei obigem Funktionsaufruf wurden Keyword-Argumente verwendet.
        # https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments
        #
        # Verwenden Sie nun eine list comprehension, um die Elemente der Liste rand_list
        # ganzzahlig zu runden und zu einem int zu konvertieren.
        # https://docs.python.org/3/library/functions.html#round
        # https://docs.python.org/3/library/functions.html#int
        # Verwenden sie dann ein dict Object, um zu zaehlen wie oft jeder int in der 
        # Liste auftritt. Als Schluessel verwenden Sie dabei den int, als Wert die 
        # entsprechende Anzahl.
        # Sie bestimmen also ein Histogramm, das die empirische Verteilung der Daten
        # darstellt.
        #
        # Geben Sie anschliessend das dict Objekt nach Schluessel sortiert in der 
        # Konsole aus. Verwenden Sie die sorted() Methode. Beachten Sie dabei,
        # dass die Schluessel-Wert Paare in einem Dictionary natuerlicher Weise 
        # keine Reihenfolge haben. Verwenden zunaechst Sie eine geieignete Funktion 
        # um aus dem Dictionary eine Liste mit Schluessel-Wert tuple Objekten zu
        # erzeugen.
        # https://docs.python.org/3/howto/sorting.html#sortinghowto

        hist = {}
        print(type(hist))
        rounded = [int(round(x)) for x in rand_list]
        for x in rounded:
            if x not in hist:
                hist[x] = 0
            hist[x] += 1

        # Ein fuer die Berechnung von Histogrammen besser geeignetes Dictionary bietet
        # das defaultdict
        # https://docs.python.org/3/library/collections.html#collections.defaultdict
        #
        # Beim defaultdict gibt man bei der Initialisierung eine sogenannte default_factory
        # als Parameter mit. Damit werden default Werte fuer Schluessel, die noch nicht in
        # dem defaultdict vorhanden sind, automatisch erzeugt. Die int Funktion
        # erzeugt zum Beispiel den Wert 0. Schauen Sie sich dazu auch die Beispiele auf
        # der oben genannten Webseite an.
        #
        # Berechnen Sie nun das Histogramm ueber die gerundeten Werte aus rand_list 
        # mit einem defaultdict. 

        dhist = defaultdict(int)
        for x in rounded:
            dhist[x] += 1

        # Wenn man ermitteln moechte, welcher Schluessel in einer gegebenen Menge
        # von Werten am haeufigsten ist, muss man in dem Histogramm den Schluessel zu
        # den groessten Wert bestimmen. 
        # In Analogie zu der Sortierung von Schluessel-Wert Paaren (siehe oben)
        # nach Schluessel, sollen Sie nun nach Wert sortieren und den Schluessel
        # dem groessten Wert in einer Variable speichern und ausgeben. 
        # Schauen Sie sich dabei nochmal die Dokumentation der sorted Funktion
        # an und passen Sie den Aufruf im Vergleich zu oben passend an.
        # Hinweis: Sie koennen dazu die itemgetter Funktion als key Funktion verwenden.
        # https://docs.python.org/3/howto/sorting.html#key-functions

        sorted_entries = sorted(list(hist.items()), key=itemgetter(1))
        max_key = sorted_entries[-1][0]

        # Geben Sie das defaultdict Objekt sowie den Schluessel zum groessten Wert
        # als Ergebnis der Methode zurueck.

        return dhist, max_key

class NumPyIntro(object):

    #
    # SciPy / NumPy
    #
    # http://www.scipy.org/
    # http://www.scipy.org/docs.html
    # http://www.numpy.org/
    # https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

    # Sehr grosse Datenmengen, lassen sich iterativ nicht effizient in Python
    # verarbeiten. Dazu gibt es mit NumPy / SciPy eine Bibliothek, mit der 
    # numerische Berechnungen vektorisiert effizient durchgefuehrt werden koennen.
    # Vektorisiert bedeutet dabei, dass Operationen auf grossen Datenmengen direkt
    # mit einzelnen Methodenaufrufen durchgefuehrt werden und nicht ueber 
    # Schleifen in Python Code umgesetzt werden.
    #
    # ACHTUNG: diese Form der Programmierung unterscheidet sich DEUTLICH von dem
    # was sie vielleicht aus C / C++ oder auch Java gewoehnt sind.
    #
    # ------------------------------------------------------------
    # VERWENDEN SIE BEI *KEINER* DER FOLGENDEN AUFGABEN SCHLEIFEN!
    # ------------------------------------------------------------
    #

    def arrays(self, test_arr=None):
        print('\n[NumPyIntro::arrays]')
        # Elementare Datenstruktur ist das ndarray. Ein Array hat dabei eine 
        # festgelegte Anzahl von Dimensionen (ndarray.ndim) und eine festgelegt 
        # Anzahl von Elementen je Dimension (ndarray.shape).         
        # Ein neues ndarray erstellt man zum Beispiel aus einer Python Liste. 
        # Die Listenstruktur gibt dann die Struktur des Arrays vor:
        arr = np.array([1, 2, 3])
        print(arr)
        print('type: %s' % type(arr))
        print('ndim: %d' % arr.ndim)
        print('shape: (%d,)' % arr.shape)
        # Beachten Sie, dass arr.shape in diesem Beispiel ein tuple ist, in dem 
        # fuer jede Dimension die Anzahl von Elementen gespeichert wird. 
        #
        # Bei der Initialisierung von ndarrays mit Listen lassen sich weitere
        # Dimensionen durch Listen von Listen darstellen:
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        print(arr)
        print('ndim: %d' % arr.ndim)
        print('shape: (%d, %d)' % arr.shape)
        # Wenn man in einem solchen Array eine Matrix repraesentieren moechte 
        # entspricht die erste Dimension (axis=0) den Zeilen und die zweite
        # Dimension (axis=1) entspricht den Spalten.
        print('Zeilen: %d' % arr.shape[0])
        print('Spalten: %d' % arr.shape[1])
        if test_arr is None:
            test_arr = arr

        # Es gibt auch Funktionen um Arrays mit einer bestimmten Struktur und
        # einem initialen Wert zu erstellen. Zum Beispiel mit Nullen oder Einsen.
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html
        zeros_arr = np.zeros((3, 3))
        print(zeros_arr)

        # Die Struktur eines ndarray koennen Sie mit reshape aendern
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
        # 
        # Erzeugen Sie nun ein Array, das eine Sequenz von Zahlen zwischen 
        # 5 und 17 (ausschliesslich) enthaelt und bringen Sie es in die Form einer
        # 4x3 Matrix. 
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
        #
        # Verwenden Sie die numpy Funkionen reshape und arange. Speichern Sie 
        # das Ergebnis in der Variable arr.

        arr = np.arange(5, 17).reshape((4, 3))

        # Mit reshape koennen Arrays mit mehreren Dimensionen auch linearisieren
        # Dabei muessen Sie beachten, ob entlang der Zeilen oder Spalten linearisiert
        # wird. 
        # Linearisieren Sie das ndarray test_arr zu der Dimension 1, entlang
        # der Zeilen und dann auch entlang der Spalten. Verwenden Sie die Variablen   
        # arr_lin_rows und arr_lin_cols respektive.
        # Lesen Sie dazu zunaechst die Hilfeseite der reshape Funktion:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
        #
        # Hinweise: 
        #  - Ein ndarray laesst sich durch Verwendung des newshape Parameters 
        #    mit Wert -1 linearisieren. 
        #  - Schauen Sie sich den Parameter order genau an. 
        #
        # Geben Sie die Ergebnisse aus. Die Variablen werden als Ergebnis der
        # Funktion zurueckgegeben. 

        arr_lin_rows = test_arr.reshape(-1, order="C")
        arr_lin_cols = test_arr.reshape(-1, order="F")

        return arr, arr_lin_rows, arr_lin_cols

    def array_access(self, seq_arr=None):
        print('\n[NumPyIntro::array_access]')

        # Im folgenden schauen wir uns Zugriffsfunktionen fuer ndarrays an.

        if seq_arr is None:
            seq_arr = np.arange(100).reshape(10, 10)

        print(seq_arr.shape)
        print(seq_arr)

        # Geben Sie das Element in der dritten Zeile und fuenften Spalte als
        # Ergebnis der Funktion zurueck. Zusaetzlich koennen Sie das Ergebnis
        # ausgeben.
        #
        # Hinweis: Schauen Sie sich dazu auch das Numpy Quickstart Tutorial an:
        # https://docs.scipy.org/doc/numpy/user/quickstart.html#indexing-slicing-and-iterating
        return seq_arr[2, 4]

    def array_slicing(self, seq_arr=None):
        print('\n[NumPyIntro::array_slicing]')

        if seq_arr is None:
            seq_arr = np.arange(100).reshape(10, 10)
        # Slicing: 
        #    Geben Sie die obere (3,2) Matrix und die untere (3,3) Matrix aus.
        #    Geben Sie die 4. Zeile aus.
        #    Geben Sie die jede ungerade Spalte aus, also Spalten 1,3,.. Verwenden 
        #        sie dabei slicing mit step=2.
        #
        # Speichern Sie alle Zwischenergebnisse in Variablen und geben Sie diese
        # aus und in der genannten Reihenfolge als Ergebnis der Funktion zurueck.
        # 
        # Hinweis: Schauen Sie sich dazu auch das Numpy Quickstart Tutorial an:
        # https://docs.scipy.org/doc/numpy/user/quickstart.html#indexing-slicing-and-iterating

        return seq_arr[0:3, 0:2], seq_arr[-3:, -3:], seq_arr[3], seq_arr[:, 1::2]

    def array_indexing(self, seq_arr=None):
        print('\n[NumPyIntro::array_indexing]')

        if seq_arr is None:
            seq_arr = np.arange(100).reshape(10, 10)

        # Integer Array Indexing:
        # Um auf mehrere Elemente zuzugreifen sind Index Arrays hilfreich.
        # Dabei wird eine Menge gewuenschter Array Elemente direkt durch ihre 
        # Index Positionen angesprochen.
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing

        #    Geben Sie die Elemente (1,3), (2,1), (3,1) mit Hilfe von Index
        #        Arrays aus.  
        #    Index arrays lassen sich auch mit Slicing Operationen kombinieren
        #        Geben Sie alle Zeilen und die Spalten mit Index 2,3,6 aus.

        one = seq_arr[[1, 2, 3], [3, 1, 1]]
        two = seq_arr[:, [2, 3, 6]]

        # Boolean Indexing:
        # Boolsche Index Arrays funktionieren als Masken, bei der an jeder Array
        # Position ein boolscher Wert gespeichert ist, der angibt ob, das 
        # entsprechende Element in dem zu indizierenden Array ausgewaehlt ist 
        # oder nicht. Dies eignet sich hervorragend um Array Elemente zu bestimmen
        # auf die eine bestimmte Bedingung zutrifft.
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing
        #
        # Die folgenden beiden Aufgaben bauen aufeinander auf:
        #     Setzen Sie nun alle ungeraden Elemente des ndarray seq_arr auf 0 
        #         und geben Sie das Ergebnis aus. 
        #     Geben Sie dann alle Elemente zurueck die groesser als 17 und 
        #         ganzzahlig durch drei teilbar sind.
        #         http://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_and.html

        seq_arr[seq_arr % 2 != 0] = 0
        four = seq_arr[np.logical_and(seq_arr > 17, seq_arr % 3 == 0)]

        # Speichern Sie die vier Egebnisse jeweils in ein Array und geben Sie diese 
        # als Ergebnis der Funktion zurueck.

        return one, two, seq_arr, four

    def array_operations(self, seq_arr=None):
        print('\n[NumPyIntro::array_operations]')

        if seq_arr is None:
            seq_arr = np.arange(100).reshape(10, 10)

        # Mathematische Operation auf Arrays werden in erster Linie elementweise
        # durchgefuehrt. Fuer eine Matrixmultiplikation, also fuer Skalarprodukte,
        # verwendet man die dot Funktion.
        #
        # Fuehren Sie die folgenden Opertionen auf dem Array seq_arr durch.
        #    Geben Sie die Summe der ersten beiden Zeilen aus (elementweise).
        #    Geben Sie das Produkt der ersten beiden Zeilen aus (elementweise).
        #    Geben Sie das Skalarprodukt der ersten beiden Zeilen aus.
        # Verwenden Sie dazu die Operatorn +, *  und np.dot in geeigneter Weise
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html

        one = seq_arr[0] + seq_arr[1]
        two = seq_arr[0] * seq_arr[1]
        three = np.dot(seq_arr[0], seq_arr[1])

        # Speichern Sie alle Ergebnisse und geben Sie diese als Ergebnis der
        # Funktion zurueck.

        return one, two, three

    def array_functions(self, seq_arr=None):
        print('\n[NumPyIntro::array_functions]')

        if seq_arr is None:
            seq_arr = np.arange(100).reshape(10, 10)
        #
        # Nun schauen wir uns noch einige wichtige NumPy Funktionen an.
        #

        # Bestimmen Sie in dem ndarray seq_arr in jeder Zeile das groesste Element
        # und seinen Index. Geben Sie die Ergbnisse aus. Speichern Sie das Array
        # mit Indices gefolgt von dem Array mit Werten in einem Tuple.
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html

        one = (np.argmax(seq_arr, axis=1), np.amax(seq_arr, axis=1))

        # Bestimmen Sie die Summen ueber die Elemente jeder Zeile.
        # Speichern Sie das Ergebnis in einer Variable.

        two = seq_arr.sum(axis=1)

        # Teilen Sie nun jedes Element des Arrays durch die Summe der entsprechenden
        # Zeile. Eine solche Operation laesst sich als elementweise Division zwischen
        # Arryays ausdruecken bei der sich Bezug zwischen Divdend und Divisor
        # durch den Zeilen bzw Spalten Index der Elemente in den Arrays ergibt.
        #
        # Vorsicht: Achten Sie bei Divisionen auf die Datentypen der Arrays. Hier
        # sollen sie eine Floating-Point Division durchfuehren. 
        #
        # Hinweise:
        # - Die elementweise Zuordnung zwischen Zeilen (Dividend) und Zeilensummen
        #   (Divisor) ergibt sich durch den shape der ndarrays. 
        # - Fuer die elementweise Division verwenden Sie den / Operator.
        #
        # Geben Sie die Zwischenergebnisse aus und inspizieren Sie diese auch
        # mit dem Debugger.

        three = seq_arr / two[:, None]

        # Speichern Sie alle Ergebnisse und geben Sie diese als Ergebnis der
        # Funktion zurueck.

        return one, two, three

    def array_distributions(self):
        print('\n[NumPyIntro::array_distributions]')
        #
        # Zum Abschluss schauen wir uns noch ein etwas komplexeres Beispiel
        # inklusive Visualisierung an.
        #
        # Die folgenden Arrays enthalten normalverteilte und gleichverteilte Zufalls-
        # zahlen.

        rand_arr_gen = RandomArrayGenerator()

        rand_arr_gauss = rand_arr_gen.rand_gauss(arr_shape=(10000,), mean=50, std_deviation=10)
        rand_arr_unif = rand_arr_gen.rand_uniform(arr_shape=(10000, 50), min_elem=0.5, max_elem=10.5)

        # Runden Sie die Elemente der Arrays (ganzzahlig) und erstellen sie jeweils 
        # Histogramme. Um das Histogramm fuer rand_arr_unif zu erstellen, linearisieren
        # Sie die Matrix zunaechst. (Verwenden Sie reshape mit shape Parameter -1.)
        # Das Histogramm koennen Sie dann mit der NumPy Funktion bincount berechnen.
        # Beachten Sie, dass die bincount Funktion ein Integer Array als Eingabe
        # benoetigt. Warum ist das so?
        # Plotten Sie die Ergebnisse mit matplotlib. Verwenden Sie dazu die Methode 
        # bar_plot (unten). Da Sie im weiteren Verlauf der Uebung weitere Plots
        # erstellen muessen, schauen Sie sich die Methode bar_plot an.
        # http://matplotlib.org/
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

        gauss_histogram = np.bincount(np.array(np.round(rand_arr_gauss), dtype="int").reshape(-1))
        unif_histogram = np.bincount(np.array(np.round(rand_arr_unif), dtype="int").reshape(-1))

        bar_plot(None, gauss_histogram)
        bar_plot(None, unif_histogram)

        # Bilden Sie nun die zeilenweisen Summen ueber das ndarray mit den gleichverteilten
        # Zufallszahlen (rand_arr_unif). Berechnen und visualisieren Sie das Histogramm 
        # wie zuvor. Erklaeren Sie das Ergebnis.
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
        #
        # Geben Sie das Ergebnis als Rueckgabe der Funktion zurueck.

        unif_histogram = np.bincount(np.array(np.round(rand_arr_unif), dtype="int").sum().reshape(-1))
        bar_plot(None, unif_histogram)


def bar_plot(x_values, y_values, title=None):
    """ Plottet ein vertikales Balkendiagramm
    Params:
        x_values: Liste von x Werten. Auf None setzen, um den Index aus y_values
            zu verwenden. (Automatische Anzahl / Platzierung der x-ticks).
        y_values: Liste von y Werten
        title: Ueberschrift des Plots
    """
    x_pos = np.arange(len(y_values))

    # plt.figure() erzeugt ein Figure-Objekt. Eine Figure ist ein Container, der
    # die Plot-Elemente enthaelt.
    #
    # fig.add_subplot(r, c, i) gibt ein Axes-Objekt aus der Figure fig. r und c
    # definieren ein Grid von Axes mit r Reihen und c Spalten. Das Axes-Objekt
    # mit Index i soll zurueckgegeben werden. Der Index beginnt bei 1 und erhoeht
    # sich zuerst entlang der Reihen.
    # Wenn r, c und i einstellig sind, kann auch ein dreistelliger Parameter genutzte werden.
    # Bsp: add_subplot(3, 2, 1) entspricht add_subplot(321).
    #
    # Mit plt.show() werden alle bisher erzeugten Figure-Objekte angezeigt.
    #
    # Siehe auch:
    # http://matplotlib.org/api/figure_api.html
    # http://matplotlib.org/api/axes_api.html

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_pos, y_values, width=0.9, align='center', alpha=0.4)
    if x_values is not None:
        ax.set_xticks(np.linspace(0, len(y_values), len(x_values)))
        ax.set_xticklabels(x_values, rotation='vertical')
    if title is not None:
        ax.set_title(title)
    plt.show()


def intro():
    #
    print('Willkommen beim Fachprojekt Dokumentenanalyse!')
    #
    # In diesem "intro" Modul sollen Sie sich mit den grundlegenden 
    # Eigenschaften und Funktionen von Python, NumPy/SciPy und matplotlib
    # vertraut machen. Die Aufgaben sind dazu gedacht Ihnen den Einstieg
    # zu erleichtern. Weiterfuehrende Informationen sind auf der Webseite
    # des Fachprojekts verlinkt. 

    # WICHTIG: Lesen Sie die Kommentare sorgfaeltig! Oft finden sich Hinweise
    # zu Python / NumPy / SciPy Funktionen, die Ihnen die Arbeit erheblich
    # erleichtern
    #
    # Fuehren Sie die Unit-Tests in test.intro_test aus, um ihre Ergebnisse zu
    # ueberpruefen.
    # WICHTIG: Die Unit-Tests werden die von Ihnen implementierten Methoden
    # mit alternativen Eingaben ausfuehren. Achten Sie daher darauf Loesungen
    # zu finden, die nicht nur fuer die per Default gegebenen Daten, sondern
    # im Allgemeinen funktionieren!
    #
    # Falls Unit-Tests fehlschlagen, obwohl Sie der Meinung sind, die Aufgaben
    # entsprechend der Aufgabenstellung umgesetzt zu haben, schauen Sie sich die
    # im Unit-Test erwartete Loesung an. Ueberlegen Sie, wie die im Unit-Test
    # vorgesehnen Loesungen mit der Aufgabenstellung in Beziehung gebracht werden
    # koennen.
    # Lesen Sie die Aufgabenstellung in solchen Faellen nochmals sehr genau! 

    py_intro = PythonIntro()
    py_intro.datatypes()
    py_intro.sequences()
    py_intro.sequences_complex()
    py_intro.list_comprehension()
    py_intro.dictionaries()

    np_intro = NumPyIntro()
    np_intro.arrays()
    np_intro.array_access()
    np_intro.array_slicing()
    np_intro.array_indexing()
    np_intro.array_operations()
    np_intro.array_functions()
    np_intro.array_distributions()


if __name__ == '__main__':
    intro()
