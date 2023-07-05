# Endometrical cancer segmentation
Objective of reproducing the reported results in [Automated segmentation of endometrial cancer on MR images using deep learning](https://link.springer.com/content/pdf/10.1038/s41598-020-80068-9.pdf)-

The trained weights and exported learner available on [HugginFace](https://huggingface.co/skaliy/endometrical_cancer_segmentation). 

## How to use
1. Clone this repository.
2. Install fastMONAI by following the instructions provided [here](https://github.com/MMIV-ML/fastMONAI/tree/master).
3. (<b>Optional</b>) Run the `01_ec_training.ipynb` notebook to train your own model.
4. Run the `02-ec-inference.ipynb` or `inference_script.py` to perform inference with the trained model.
If you choose to use `inference_script.py`, please follow these steps:

- Make the script executable using the following command in the terminal: `chmod +x inference_script.py`
- Run the script by executing the following command in the terminal: `python inference_script.py IMG_PATH`

## Results
The box plot of the predictions on the validation set: 
![](figs/boxplot.png)

The results from the validation set are also presented in the table below:

| subject_id | tumor_vol | inter_rater | r1_ml    | r2_ml    |
|------------|-----------|-------------|----------|----------|
| 29         | 4.16      | 0.201835    | 0.806382 | 0.006231 |
| 32         | 8.00      | 0.684142    | 0.293306 | 0.209449 |
| 36         | 19.06     | 0.928750    | 0.793611 | 0.785335 |
| 47         | 11.01     | 0.944209    | 0.905159 | 0.902548 |
| 50         | 6.26      | 0.722867    | 0.619272 | 0.631579 |
| 65         | 13.09     | 0.930613    | 0.879279 | 0.850546 |
| 67         | 3.71      | 0.943498    | 0.887189 | 0.878163 |
| 75         | 7.16      | 0.263539    | 0.774411 | 0.266463 |
| 86         | 7.04      | 0.842577    | 0.821208 | 0.798148 |
| 135        | 8.10      | 0.839964    | 0.758176 | 0.680348 |
| 140        | 19.78     | 0.895506    | 0.936177 | 0.874019 |
| 164        | 16.98     | 0.905008    | 0.923559 | 0.887268 |
| 246        | 6.59      | 0.899448    | 0.907503 | 0.871254 |
| 255        | 36.22     | 0.955784    | 0.927517 | 0.921816 |
| 343        | 0.69      | 0.528261    | 0.840237 | 0.600751 |
| 349        | 2.96      | 0.912664    | 0.828181 | 0.778983 |
| 367        | 1.02      | 0.073485    | 0.392027 | 0.117796 |
| 370        | 10.82     | 0.953443    | 0.917094 | 0.908893 |
| 371        | 3.83      | 0.859781    | 0.685033 | 0.618380 |
| 375        | 11.67     | 0.911141    | 0.921345 | 0.910804 |
| 377        | 4.37      | 0.782994    | 0.712791 | 0.680165 |
| 381        | 7.63      | 0.891990    | 0.245768 | 0.237990 |
| 385        | 2.67      | 0.803215    | 0.641916 | 0.601690 |
| 395        | 0.68      | 0.770738    | 0.204418 | 0.242908 



## Support and Contribution
For any issues related to the model or the source code, please open an issue in the corresponding GitHub repository. Contributions to the code or the model are welcome and should be proposed through a pull request.
