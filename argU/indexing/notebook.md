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

## Das Debatten-Embedding

Insgesamt gibt es 59.637 verschiedene Debatten in dem Datensatz.

## Säuberung

Argumente die weniger als 25 Terme haben scheinen sinnlos zu sein! Man sollte sie löschen können. Beispiele:

* Uhhh, please vote con. It seems my opponent has not been able to respond in time. Vote con :)
* Vote Pro!
* My arguments still stand.
* Fast food is not that bad you have the choice of a salad or a hamburger
* I have a losing streak almost in the twenties while you have a mear 5. . come back when you earn your fail.
* Prove me wrong.
* My opponet as found no way to counter myargument because its obviously true.
* no no
* Its unbelievable how stupid people are today, especially in America.
* Extend my previous arguments into this Round. I request that the voters do not penalize my opponent for the forfeit.
* I would like to keep this round for acceptance only. Thanks, DDD
* Vote Pro!
* Arguments Extanded
* Arguments Extanded.
* Arguments Extended.
* Atheists believe in mythological concepts. Good luck.
* Blake is a good player but not the best, as arguments show, and he has not posted any arguments, vote con.
* Vote Con
* LOOK at his states if you want to argue!)
* I accept
* racist mofo
* No
* I like animals, so it is not morally sound.
* rape is sexy also do you play roblox
* Accepted
* And so i signs the contract NO VOTING FOR YOURSELF Btw, my choice of gun is the G18C (or G 18) Your arguments please, XD
* I wish him well. I am sure he is busy with his studies.
* Con forfeit <NUM> rounds so far. vote pro.
* con
* v
* forfeit is not a race, i am the last to forfeit in this debate to last comes after first i did forfeit
* i forfeit
* I think the title of the debate speaks for itself. Good luck to Masterful/What50, you'll need it cunt.
* True. Although you don't need to leave, you could try logging on while sober and see how that works out for you?
* Ohhh Masterful - I showed you my dick. Answer me!


BM25: Model_texts
CBOW: Model_texts
A2V: Model_texts -> Jedes Word sucht Embedding aus CBOW