# KINS-experiments (CILC 2022)
KINS - Knowledge Injection via Network Structuring experiments (CILC 2022).

## Setup

You can execute a list of predefined commands by running:
`python -m setup.py commandName`.

### Experiments
Write `python -m setup.py run_kins_experiment` to launch a set of experiments using KINS on the splice junction dataset.
Default options are:
- `-m=fold`, 10-fold cross validation;
- `s=0`, random seed;
- `f=result`, file name to save the results.