# UPP2DigraphSearch

Collection of Python functions for researching digraphs with <u>U</u>nique <u>P</u>ath <u>P</u>roperty of length 2, more commonly called central digraphs.

## Description

Repository for the source code for my (Andrey Lugovskoy) Honours project. Contains multiple important functions for researching central digraphs (CDGs), digraphs whose adjacency matrix satisfies $A^2=J$ where $J$ is the all 1's matrix.
The structure of the project is as follows:

* The file `src/genUPPDigraphs.py` contains methods for generating central digraphs through brute-force approaches.
* The file `src/sageGAP.py` loads a running instance of GAP through a SageMath install, for the purposes of speeding up an algorithm in `src/genUPPDigraphs.py`.
Due to this, both this and the above file must be run from the internal version of python of SageMath, as will be described in Execution section below.
* The file `src/knuthianUPP.py` contains methods for generating and manipulating *Knuthian* central digraphs (described in Theorem 2 of [this paper](https://doi.org/10.1016/S0021-9800(70)80032-1)).
It further has functions for dealing with multiplication tables inherent to this family of CDGs, such as their explicit isomorphism check.
* The file `src/switchingsUPP.py` contains methods for performing switches as described by Fletcher in [this paper](https://www.proquest.com/dissertations-theses/unique-path-property-digraphs/docview/303962591/se-2).
Included is a method for finding the switching equivalence class of a given central digraph.
* The files `src/outputs.py` and `src/readDataFiles.py` are concerned with writing and reading important data to the `data/` directory, to avoid expensive recompution.
Most notable of this is the file `data/orderedComplete4`, the first published (otherwise credit to Kündgen, Leander and Thomassen in [paper](https://doi.org/10.1016/j.jcta.2011.03.009)) list of all non-isomorphic 3492 CDGs on 16 vertices.
The list is arranged in decreasing lexicographical order of the CDGs where each adjacency matrix is converted into a binary string of length 256 by concatenating the rows of the adjacency matrix. 
* The file `src/uncatExperiments.py` contains esoteric functions and personal notes about properties of CDGs that I have researched. Used mainly as draft and playing ground for finding conjectures to research.
* The file `src/utilsUPP.py` is a library of utility functions including common methods used on CDGs, as well as functions which link the different mathematical Python libraries used.

## Getting Started

### Dependencies

Languages used:

* Python    Version 3.10.12 (downgraded to version 3.9.9 by SageMath)

Python dependencies are listed in `requirements.txt`.

Make sure to have the following programs installed, and make sure their executables are in path, so that they can be called from command line.

* GAP       Version 4.12.2
* SageMath  Version 9.5
* Minion    Version 2

Developed and tested on Ubuntu 22.04.3 LTS, has not been tested on other operating systems.

### Installation

Make sure a Python package manager is installed, below assumes `pip` is used.

Install the necessary packages by running

```bash
pip install -r requirements.txt
```

Install [GAP](https://www.gap-system.org/Releases/), [SageMath](https://github.com/sagemath/sage) and [Minion](https://github.com/minion/minion) from their respective pages and follow the installation instructions provided there.
Add the directories containing their executables to path.

### Execution

Once all python packages are installed, the source files can be run by invoking python with the file name. 

The exceptions to this are the files `src/genUPPDigraphs.py` and its dependency `src/sageGAP.py` which must run from SageMath. To do this first make sure the internal Python of SageMath has all necessary depencies by running

```bash
sage -pip install -r requirements.txt
```

and then invoke either of the two above files with SageMath's internal Python by running

```bash
sage -python [file.py]
```

## Authors

* Andrey Lugovskoy -- University of Western Australia
* Gordon Royle (Advisor) -- University of Western Australia

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
