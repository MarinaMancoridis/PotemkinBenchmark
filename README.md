# Potemkin Benchmark Documentation

Source code for the paper "[Potemkin Understanding in Large Language Models](https://arxiv.org/abs/2506.21521)" by Marina Mancoridis, Keyon Vafa, Bec Weeks, and Sendhil Mullainathan.

If you find this repository useful for your research, please consider citing our work:

```
@article{mancoridis2025potemkin,
  title={Potemkin Understanding in Large Language Models},
  author={Mancoridis, Marina and Weeks, Bec and Vafa, Keyon and Mullainathan, Sendhil},
  journal={arXiv preprint arXiv:2506.21521},
  year={2025}
}
```


## Introduction

This guide is structured into the following main components:

* **Installation**
* **Quickstart**
* **Benchmark Dataset**
* **Automatic Evaluation**
* **Incoherence**


Below, you'll find detailed instructions to effectively use each component.

## Installation

Before you begin, make sure you have [Conda](https://docs.conda.io/) (version ≥4.6) installed on your system.

1. **Clone the repository**

   ```bash
   git clone https://github.com/MarinaMancoridis/PotemkinBenchmark.git
   cd PotemkinBenchmark
   ```

2. **Create the Conda environment**

   We provide an `environment.yml` file listing all required packages (Python 3.9+). To create the environment, run:

   ```bash
   conda env create --file environment.yml
   ```

3. **Activate the environment**

   ```bash
   conda activate potemkin
   ```

## Quickstart

Get up and running in a few simple steps:

1. **Run a sample command**: This command reproduces Table 1, showing potemkin rate by task.

   ```bash
    cd BenchmarkDataset && python -c "from potemkin_rates import print_potemkin_rate_by_task; print_potemkin_rate_by_task()"
   ```

2. **View labeled instances**

    The file `BenchmarkDataset/labeled_instances.csv` contains labeled instances and non-instances of each concept, across all three domains. These labeled instances are used in subtasks of the benchmark dataset analysis.

## Benchmark Dataset

The `BenchmarkDataset` directory is organized by the four main tasks in our framework, each contained within its own subdirectory:

* **Define**
* **Classify**
* **Generate**
* **Edit**

### Reproducing Table 1

To reproduce **Table 1** (potemkin rates by task), run the following command from the repository's root directory:

```bash
cd BenchmarkDataset && python -c "from potemkin_rates import print_potemkin_rate_by_task; print_potemkin_rate_by_task()"
```

### Accessing the Data

* We provide an iterator to access labeled model responses for each task in a standardized format.
* A sample API for using the iterators is provided in `BenchmarkDataset/main.py`. 
* The source code for the iterators themselves can be accessed in `BenchmarkDataset/iterators.py`.
* Helper functions to compute potemkin rates can be found in `potemkin_rates.py`.
* Additional configuration details, such as the lists of models and concepts, are defined in `constants.py`.

## Automatic Evaluation

For the automatic evaluation, we use the `AutomaticEval` directory. Make sure to set up the API keys in the `private/models.py` file; you can do this by running `export OPENAI_API_KEY=...` and so on.

To run the automatic evaluation, go to the `AutomaticEval` directory and run
```
python main.py
```

The results will be saved in the `AutomaticEval/results` directory.

## Incoherence

All relevant results for our incoherence analysis are provided in the `Incoherence` directory. 

### Reproducing the first column of Table 2

To reproduce the first column of **Table 2** (incoherence rates by model), run the following command from the repository's root directory:

```bash
cd Incoherence && python -c "from incoherence_rates import print_incoherence_by_model; print_incoherence_by_model()"
```

### Reproducing Table 2

To emit the LaTeX for all of **Table 2**, run `python table_2.py` from the repository's root; this script recomputes the incoherence column and uses the published, hard-coded potemkin lower-bound values, which are produced on the fly by the Automatic Evaluation procedure above.
