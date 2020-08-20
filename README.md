# fbd-interpreter Package
The interpreter package of the Fab Big Data (SNCF).

Its purpose is to make state-of-the-art machine learning and deep learning interpretability techniques easy to use. 

## Live Demo

## Requirements 
**fbd_interpreter** 1.0 requires : 
* Python 3.6+

Before creating a virtual environment, make sure Python3.6 is installed. If not, install it with `sudo apt-get install python3.6-dev`.

## Installation instructions

Create a virtual environment, by default in .venv directory: `virtualenv -p python3.6 .venv`.

Source the virtual environment with `source .venv/bin/activate`.

Install Python dependencies with `pip install -r config/requirements.txt`

Install project modules with `pip install --editable .`, this command runs the `setup.py` script to make the package `fbd_interpreter` available in the environment.

## Pre-requisites 

Update required  configuration variables located in `fbd_interpreter/config/config_local.cfg` 

## Features

This package incorporates state-of-the-art machine learning **interpretability techniques** under one roof. 

With this package, you can understand and explain your model's global behavior **global interpretability**, understand the reasons behind individual predictions **local interpretability** or both (mix).

The techniques available in this package are:

### Global interpretability
- Partial Dependecy Plots (from icecream)
- Individual Conditional Expectation Plots (from icecream)
- Accumulated Local Effects Plots (from icecream)
- SHAP plots for feature importance and impact (from SHAP)
### Local interpretability
- SHAP plots for local explanation with TreeBased model (from SHAP)
- SHAP plots for local explanation with non TreeBased model (from SHAP)

Details information are given [here](https://wiki-big-data-ia.intranet.itnovem.com/index.php/REaDME_:_Extraire_la_logique_m%C3%A9tier)

### Supported techniques 

| Interpretability Technique | Interpretability Type | Model Type |
| --- | --- | ---|
|ShapTreeExplainer - Feature Importance|Global|Tree Based models|
|ShapKernelExplainer - Feature Importance|Global|Non Tree Based models|
|Partial Dependecy Plots|Global|Model Agnostic|
|Individual Conditional Expectation Plots|Global|Model Agnostic|
|Accumulated Local Effects Plots|Global|Model Agnostic|
|ShapTreeExplainer - force plot|Local|Tree Based models|
|ShapKernelExplainer - force plot|Local|Non Tree Based models|



## Quickstart

### Usage via command line interface [CLI] 
The most straightforward way is to use the **interpret** click command , that wraps around most of the functionality in the module.

A basic usage example is shown below :

```bash src
python fbd_interpreter/main.py 

```

Supported parameters are :

```bash src
python fbd_interpreter/main.py --help

Usage: main.py [OPTIONS]

Options:
  --interpret-type   Type d'interprétabilité: Choisir global, local ou mix
                     [default: global]

  --use-ale          Calculer et afficher les plots ALE  [default: True]
  --use-pdp-ice      Calculer et afficher les plots PDP & ICE  [default: True]
  --use-shap         Calculer et afficher les plots de feature importance SHAP
                     [default: False]

  --help             Show this message and exit.

```

### Usage as external module  [Python package] 

One way of using the package is  to run the `interpret` function which takes care  of explaining model behaviour .

For instance , using **partial dependency plots** for global interpretability:
```python
from fbd_interpreter.main import interept
interept(interpret_type="global",use_pdp_ice=True,use_ale=False,use_shap=False)
```

## Test



## Deployment 

In order to configure deployment environment , one may create environment variable `INTERPRET_ENV`
to specify deployment env , two  modes are supported :
- Deploy in dev env
- Deploy in prod env 

By default , `INTERPRET_ENV = "local"` 

To create new deployment modes :
- Update `INTERPRET_ENV` :  ```export INTERPRET_ENV = $deploy_env ```
- Create new configuration file named `config_{deploy_env}.cfg` based on existing templates 
- Copy configuration file  in `config/` directory 


## Git hooks

Copy hooks file from config/hooks/ in .git/hooks and make them executable
```
cp config/hooks/pre-commit .git/hooks/
cp config/hooks/post-commit .git/hooks/
chmod 775 .git/hooks/pre-commit
chmod 775 .git/hooks/post-commit
```
At each commit, those hooks make a clean copy (without outputs) of all notebooks from jupyter/ to notebook_git/ and commit them. If you need to use the notebook of another user, copy it from notebook_git to your folder in jupyter/ before modifying it.

## Upgrade dependencies and freeze (if necessary)

Upgrade Python dependencies with `pip install -r config/requirements-to-freeze.txt --upgrade`

Then freeze dependencies with the command `pip freeze | grep -v "pkg-resources" > config/requirements.txt` (the `grep` part deals with a bug specific to Ubuntu 16.04, see https://github.com/pypa/pip/issues/4022)


## Support 

Road map for future developements : 
- [ ] Deep Learning Interpreter
- [ ] Saliency Maps
- [ ] xx



## Linting 
Please for future development , use black for code formatting  

## References 



## Copyright 
DSE Team - Big Data Fab

## Author
* **Soumaya IHIHI** - *Data scientist*

