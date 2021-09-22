# word_embeddings_covid-19_vocab

Am Beispiel ausgewählter Wörter des Lexikons der Covid-19-Pandemie soll mithilfe von Word Embeddings lexikalischer Bedeutungswandel überprüft werden.

Die zu untersuchenden Begriffe sind: 

- Abstandsregel
- Kontaktverbot
- Maske
- Querdenker
- Corona

Die Basis des Projekts bilden gezielte Suchabfragen auf das ZDL-Regionalkorpus des DWDS (https://www.dwds.de/d/korpora/regional), ein Korpus aus Lokal- und Regionalteilen deutscher Zeitungen von 1993 bis jetzt (Stand 02.09.2021).

Anhand der Ergebnisse dieser Suchabfragen wurden eigene Korpora erstellt, auf die die Embeddings trainiert werden.

## Ordnersturktur

### Notebooks
- den Python-Code in Form von Jupyter-Notebooks für:
    - die Korpusbereinigung
    - das Trainig der Word Embeddings 
    - die Ergebnisse 
    - die Überprüfung der Ergebnisse
    - Visualisierungen 

### plots
- Visualisierungen 

### sources
- die Quellen für die Daten des DWDS

## Korpus und trainierte Modelle
Das gesamte Projekt (inklusive Korpora und trainierte Modelle) ist auf Zenodo zu finden und kann heruntergeladen werden.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5521618.svg)](https://doi.org/10.5281/zenodo.5521618)