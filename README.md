# Sparsity Makes Sense
This repo contains the code base for reproducing the experiments included in the following publications:
* [Combating the Curse of Multilinguality in Cross-Lingual WSD by Aligning Sparse Contextualized Word Representations](https://aclanthology.org/2022.naacl-main.176/)
* [Sparsity Makes Sense: Word Sense Disambiguation Using Sparse Contextualized Word Representations](https://www.aclweb.org/anthology/2020.emnlp-main.683/)

If you would like to try out the model first, you can do so using [this demo](https://www.inf.u-szeged.hu/~berendg/nlp_demos//wsd).

## Cloning the repository
```bash
git clone git@github.com:begab/sparsity_makes_sense.git
cd sparsity_makes_sense
# you might need to install certain libs previous to the pip install command
# sudo apt -y install libblas-dev liblapack-dev gfortran
pip install -r requirements.txt
mkdir -p log/results
```

## Running the experiments
First, download the training and [evaluation data from Raganato et al. (2017)](http://wwwusers.di.uniroma1.it/~navigli/pubs/EACL_2017_Raganatoetal.pdf) by invoking  
```bash
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip WSD_Evaluation_Framework.zip
rm WSD_Evaluation_Framework.zip
```
As additional training data, you can also rely on the WordNet Gloss Tagged data provided within the [UFSAC (Unification of Sense Annotated Corpora and Tools) initiative](https://github.com/getalp/UFSAC).  

### Obtaining the dense contextualized vectors
The next step is to obtain the dense contextualized representations using the [transformers library](https://github.com/huggingface/transformers/).  
```bash
TRANSFORMER_MODEL=bert-large-cased
DATA_PATH=.

python src/01_preproc.py --gpu_id 0 \
                         --transformer $TRANSFORMER_MODEL \
                         --reader SemcorReader \
                         --in_files ${DATA_PATH}/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml \
                                    ${DATA_PATH}/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml \
                         --out_dir ${DATA_PATH}/representations > log/preproc_raganato.log 2>&1 &

python src/01_preproc.py --gpu_id 0 \
                         --transformer $TRANSFORMER_MODEL \
                         --reader WordNetReader \
                         --in_files WordNet \
                         --avg-seq \
                         --out_dir ${DATA_PATH}/representations > log/preproc_wordnet.log 2>&1 &

```
Optionally, one can use the WordNet Gloss Tagged corpus as additional training data. In order to do so, this corpus needs to be preprocessed first as well:
```bash
python src/01_preproc.py --gpu_id 0 \
                         --transformer $TRANSFORMER_MODEL \
                         --reader WngtReader \
                         --in_file ${DATA_PATH}/ufsac-public-2.1/wngt.xml \
                         --out_dir ${DATA_PATH}/representations > log/preproc_wngt.log 2>&1 &
```

### Creation of the sparse contextualized vectors

This command creates the dictionary matrix and the sparse representations averaged over the last 4 layers of the hidden representations of the transformer architecture:

```bash
LAYER=21-22-23-24
K=3000
LAMBDA=0.05

python src/02_sparsify.py --in_files ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/{semcor,ALL}.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy \
                                     ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/WordNet_${TRANSFORMER_MODEL}_avg_True_layer_${LAYER}.npy \
                          --K $K --lda $LAMBDA --normalize >> log/sparsify.log 2>&1 ;
```

Optionally, the dictionary matrices used in the NAACL 2022 publication can be accessed from [this link](https://zenodo.org/record/6535075).

### Calculate the statistics for the model
In order to calculate the affinity map ![formula](https://render.githubusercontent.com/render/math?math=\Phi) based on the sense annotated SemCor dataset and the WordNet glosses (similar to [LMMS](https://github.com/danlou/LMMS)), invoke
```bash
python src/03_train.py --norm \
                       --in_files ${DATA_PATH}/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml wordnet.txt \
                       --readers SemcorReader WordNetReader \
                       --rep ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/semcor.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}_semcor.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}.npz \
                             ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/semcor.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}_WordNet_${TRANSFORMER_MODEL}_avg_True_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}.npz \
                       --out_file ${DATA_PATH}/models/${TRANSFORMER_MODEL}_semcor_wordnet_layer${LAYER}_K${K}_lda${LAMBDA} >> log/train_semcat.log 2>&1 &
```

For obtaining a baseline model, which performs the calculation of the per synset centroids relying on the dense contextualized word representations, run the below code:
```bash
python src/03_train.py --norm \
                       --in_files ${DATA_PATH}/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml wordnet.txt \
                       --readers SemcorReader WordNetReader \
                       --rep ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/semcor.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy \
                             ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/WordNet_${TRANSFORMER_MODEL}_avg_True_layer_${LAYER}.npy \
                       --out_file ${DATA_PATH}/models/${TRANSFORMER_MODEL}_semcor_wordnet_layer${LAYER} >> log/train_semcat.log 2>&1 &
```

### Evaluate the model
Evaluating the sparse contextualied word representations, run
```bash
python src/04_predict.py --reader SemcorReader \
                         --input_file ${DATA_PATH}/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml \
                         --model_file ${DATA_PATH}/models/${TRANSFORMER_MODEL}_${MODEL}_layer${LAYER}_K${K}_lda${LAMBDA}_norm.pickle \
                         --eval_repr ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/semcor.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}_ALL.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy_normTrue_K${K}_lda${LAMBDA}.npz \
                         --eval_dir ${DATA_PATH}/WSD_Evaluation_Framework/ \
                         --batch > log/results/${TRANSFORMER_MODEL}_${MODEL}_${LAYER}_${K}_${LAMBDA}.log 2>&1 &
```
As for the baseline approach, employ
```bash
python src/04_predict.py --reader SemcorReader \
                         --input_file ${DATA_PATH}/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml \
                         --model_file ${DATA_PATH}/models/${TRANSFORMER_MODEL}_${MODEL}_layer${LAYER}_norm.pickle \
                         --eval_repr ${DATA_PATH}/representations/${TRANSFORMER_MODEL}/ALL.data.xml_${TRANSFORMER_MODEL}_avg_False_layer_${LAYER}.npy \
                         --eval_dir ${DATA_PATH}/WSD_Evaluation_Framework/ \
                         --batch > log/results/${TRANSFORMER_MODEL}_${MODEL}_${LAYER}.log 2>&1 &
```

## Results

We obtained the below results when applying sparse contextualized word representations (using the hyperparameters included in this README) for the standard WSD benchmark datasets.

| Training data | SensEval2 | SensEval3 | SemEval2007 | SemEval2013 | SemEval2015 | ALL |
| --- | --- | --- | --- | --- | --- | --- |
|SemCor | 77.6 | 76.8 | 68.4 | 73.4 | 76.5 | 75.7 |
|SemCor + WordNet |77.9 | 77.8 | 68.8 | 76.1 | 77.5 |76.8| 
|SemCor + WordNet + WNGC | 79.6 | 77.3 | 73.0 | 79.4 | 81.3 | 78.8 |

### Multilingual results on XL-WSD
Evaluation results on the [Extra-Large and Cross-Lingual Evaluation Framework for Word Sense Disambiguation](https://sapienzanlp.github.io/xl-wsd/) (XL-WSD) dataset.

|language| XLMR-Large | sparse XLMR-Large |
|----|---|---|
|BG|__72.00__|70.62|
|CA|49.97|__50.85__|
|DA|80.61|__81.13__|
|DE|83.18|__84.92__|
|EN|__76.28__|74.76|
|ES|75.85|__76.88__|
|ET|66.13|__69.33__|
|EU|__47.15__|46.71|
|FR|__83.88__|81.81|
|GL|__66.28__|65.33|
|HR|72.29|__72.71__|
|HU|67.64|__69.76__|
|IT|77.66|__78.49__|
|JA|61.87|__63.33__|
|KO|__64.20__|63.62|
|NL|59.20|__60.05__|
|SL|68.36|__68.41__|
|ZH|51.62|__53.77__|
|micro avg.|66.82|__67.21__|
|micro avg-EN|65.66|__66.28__|

## How to cite?
```
@inproceedings{berend-2020-sparsity,
    title = "Sparsity Makes Sense: Word Sense Disambiguation Using Sparse Contextualized Word Representations",
    author = "Berend, G{\'a}bor",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.683",
    pages = "8498--8508",
}
```
```
@inproceedings{berend-2022-combating,
    title = "Combating the Curse of Multilinguality in Cross-Lingual {WSD} by Aligning Sparse Contextualized Word Representations",
    author = "Berend, G{\'a}bor",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.176",
    doi = "10.18653/v1/2022.naacl-main.176",
    pages = "2459--2471",
}
```
