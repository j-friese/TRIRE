# Towards Reproducibility of Interactive Retrieval Experiments

This repository contains the code, data, and results for our ECIR'25 reproducibility paper submission, 'Towards Reproducibility of Interactive Retrieval Experiments: Framework and Case Study', to ensure transparency and facilitate reproducibility of the experiments.

## Contents of this repository

|  | Description |
| --- | --- |
| [`TRIREnotebook.ipynb`](./TRIREnotebook.ipynb) | Use this notebook to rerun the evaluations |
| `data/` | Contains the data used in this work |
| [`Sessiondata.csv`](./data/driventodistraction_sessiondata.csv) | Data from the original experiment by Azzopardi et al.[^1] Averaged values for each session. |
| [`Userdata.csv`](./data/userdata_categories.csv) | Userdata from the original experiment by Azzopardi et al.[^1] Usernames and respective categories for the different characteristics. |
| `data/splits/` | Contains the random and stratified splits to rerun the evaluation |


[^1]: Azzopardi, L., Maxwell, D., Halvey, M., Claudia Hauff, C.: Driven to Distraction: Examining the Influence of Distractors on Search Behaviours, Performance and Experience. In: Proc. CHIIR ’23, pp. 83–94. (2023)