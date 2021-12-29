# `ml_indie_tools`, a collection of machine learning tools for low-resource research and experiments

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://domschl.github.io/ml-indie-tools/index.html)
[![PyPI version fury.io](https://badge.fury.io/py/ml-indie-tools.svg)](https://pypi.python.org/pypi/ml-indie-tools/)

**Note:** THIS LIBRARY IS UNFINISHED WORK-IN-PROGRESS

## Description


### `env_tools`

A collection of tools that allow moving machine learning projects between local hardware and colab instances.

#### Examples

```python
from ml_indie_tools.env_tools import MLEnv

ml_env = MLEnv(platform='tf', accelator='fastest')
```

See the [env_tools API documentation](https://domschl.github.io/ml-indie-tools/_build/html/index.html#module-env_tools) for details.

### `Gutenberg_Dataset`

Gutenberg_Dataset makes books from [Project Gutenberg](https://www.gutenberg.org) available as dataset.

This module can either work with a local mirror of Project Gutenberg, or download files on demand.
Files that are downloaded are cached to prevent unnecessary load on Gutenberg's servers.

#### Working with a local mirror of Project Gutenberg

If you plan to use a lot of files (hundreds or more) from Gutenberg, a local mirror might be the best
solution. Have a look at [Project Gutenberg's notes on mirrors](https://www.gutenberg.org/help/mirroring.html).

A mirror image suitable for this project can be made with:

```bash
rsync -zarv --dry-run --prune-empty-dirs --del --include="*/" --include='*.'{txt,pdf,ALL} --exclude="*" aleph.gutenberg.org::gutenberg ./gutenberg_mirror
```

It's not mandatory to include `pdf`-files, since they are currently not used. Please review the `--dry-run` flag.

Once a mirror of at least all of Gutenberg's `*.txt` files and of index-file `GUTINDEX.ALL` has been generated, it can be used via:

```python
from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
gd = Gutenberg_Dataset(root_url='./gutenberg_mirror')  # Assuming this is the file-path to the mirror image
```

#### Working without a remote mirror

```python
from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
gd = Gutenberg_Dataset()  # the default Gutenberg site is used. Alternative specify a specific mirror with `root_url=http://...`.
```

#### Getting Gutenberg books

After using one of the two methods to instantiate the `gd` object:

```python
gd.load_index()  # load the index of books
```

Then get a list of books (array). Each entry is a dict with meta-data:
`search_result` is a list of dictionaries containing meta-data and the actual book-text in field `text`.

```python
search_result = gd.search({'author': ['kant', 'goethe'], language=['german', 'english']})
```

Insert the actual book text into the dictionaries. Note that download count is [limited](https://domschl.github.io/ml-indie-tools/_build/html/index.html#Gutenberg_Dataset.Gutenberg_Dataset.insert_book_texts) if using a remote server.

```python
gd.insert_book_texts(search_result)

import pandas as pd
df = DataFrame(search_result)  # Display results as Pandas DataFrame
df 
```
See the [Gutenberg_Dataset API documentation](https://domschl.github.io/ml-indie-tools/_build/html/index.html#module-Gutenberg_Dataset) for details.

### `Text_Dataset`

See the [Text_Dataset API documentation](https://domschl.github.io/ml-indie-tools/_build/html/index.html#module-Text_Dataset) for details.

### `ALU_Dataset`

See the [ALU_Dataset API documentation](https://domschl.github.io/ml-indie-tools/_build/html/index.html#module-ALU_Dataset) for details.

## History

* (2021-12-26, 0.0.x) First pre-alpha versions published for testing purposes, not ready for use.

