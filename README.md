# comp0197-group

# How to set up your environment, run our code and reproduce results used in report
1. Ensure you have the "annotations" and "images" folders taken from the Oxford-IIIT Pet Data-set (https://www.robots.ox.ac.uk/~vgg/data/pets/) in your working directory.
2. The conda environment to be used is created by running the terminal command here: "conda create -n comp0197-cw2-pt -c pytorch python=3.10 pytorch=1.13 torchvision=0.14"
3. Activate the conda environment by entering into your terminal "conda activate comp0197-cw2-pt"
4. Run "conda install -c conda-forge matplotlib" in your terminal as it is required to visualise results
5. Run "python train.py" in your terminal. This will train and save 9 different models as .pt files in your working directory, and save validation results generated during training in the form of .csv files in a folder called "val_results".
6. To visualise the validation results generated during training, run "python plot_metrics.py" in your terminal. This will generate .png files saved to the "val_results_img" folder. These figures show how the different evaluation metrics change with training, and some figures show comparisons between different ratios of labeled to unlabeled data used for training, comparisons with upper and lower bounds etc.
7. To evaluate the saved models on test data, run "python test.py" in your terminal. This will generate a "test_output.txt" in your working directory, where you can see the values for the evaluation metrics for each saved model.