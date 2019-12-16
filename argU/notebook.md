# Relevante Ergebnisse für das Indexing

Hier werden alle (Zwischen)-Ergebnisse festgehalten, um den Prozess des Indexings zu beschreiben.

## Das Word-Embedding

Das erste Embedding ist ein CBOW-Modell, das auf den ersten 10.000 Argumenten trainiert wurde.
* window = 3
* size = 300
* min_count = 3

### Preprocessing

Groß- und Kleinschreibung wurde belassen, um Begriffe wie "US" und "us" unterscheiden zu können. Jedoch wurden Hyperlinks und Elemente in eckigen klammern, die meistens nur bestimmte Elemente referenzieren, entfernt.

### Beispiele

<b>drugs</b>
* immigrants, Score: 0.7884
* alcohol, Score: 0.7748
* guns, Score: 0.7694
* cigarettes, Score: 0.7665
* criminals, Score: 0.7659
* companies, Score: 0.7640
* immigration, Score: 0.7632
* workers, Score: 0.7611
* businesses, Score: 0.7606
* products, Score: 0.7514

<b>Trump</b>
* Mr, Score: 0.8880
* Clinton, Score: 0.7921
* Martin, Score: 0.7919
* Giant, Score: 0.7662
* Donald, Score: 0.7593
* Zimmerman, Score: 0.7591
* Hillary, Score: 0.7570
* George, Score: 0.7451
* Romney, Score: 0.7401
* Lincoln, Score: 0.7313

<b>abortion</b>
* punishment, Score: 0.6973
* prostitution, Score: 0.6924
* action, Score: 0.6849
* murder, Score: 0.6763
* immoral, Score: 0.6496
* marijuana, Score: 0.6492
* marriage, Score: 0.6484
* crime, Score: 0.6356
* employer, Score: 0.6169
* legal, Score: 0.6124

<b>islam</b>
* Noxion, Score: 0.8171
* nonsensical, Score: 0.8124
* determinism, Score: 0.8024
* incoherent, Score: 0.8002
* repulsive, Score: 0.8001
* empiricism, Score: 0.7978
* TRUE, Score: 0.7900
* impermissible, Score: 0.7892
* perfection, Score: 0.7862
* greatness, Score: 0.7809

## Das Argument-Embedding

Um zu bestimmen, welchem Thema die Wörter einer Query angehören, wird jedem Argument genau ein Embedding zugeordnet, das sich aus den einzelnen Word-Embeddings zusammensetzt.