### Catalog Indexing Problem Statement

### 1. Problem Statement
#### THEME:
Retail Commerce

#### PROBLEM NAME:
Catalog indexing engine

#### DESCRIPTION OF THE PROBLEM STATEMENT:
 - Merchant catalogs can be of different types, as per the category e.g. grocery, fashion, electronics, etc;
 - Each catalog will have multiple items with each item (SKU) defined using attribute key/value pairs that defines a particular aspect of the item (e.g. colour, product name, price, etc.);
 - Optimal catalog search, using structured query or unstructured text, requires an efficient indexing engine that can support either type of search for catalogs of any size;
 - Unstructured queries typically use an inverted index (e.g. elasticsearch) that facilitates efficient retrieval of documents associated with the search query;
 - Inverted index can also be used to engineer prompts for catalog LLMs.

#### SOLUTION EXPECTED & DELIVERABLES :
 - Showcase a solution that indexes catalog in situ.
 - Catalog indexing engine should support efficient retrieval from catalogs of different sizes using any combination of unstructured & structured queries;
 - Catalog indexing engine may use indexing layer such as inverted index (e.g. Lucene-derived indexes), catalog LLMs, etc, with bidirectional integration with the original catalog source.
 - Provide a mechanism to measure the throughput of catalog indexing (for e.g. catalog record indexed per min).
 - All artefact used should be elaborated.
 - All assumptions should be elaborated.
