import re
import csv
import os

def txt_to_csv(input_txt, output_csv):
    # Read .txt file
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    # Extract data
    data = []

    for line in lines:
        if "After Epoch" in line:
            epoch = int(re.search(r'Epoch (\d+)', line).group(1))
            val_loss = re.search(r'Validation loss = (nan|[\d.]+)', line).group(1)
            iou_score = re.search(r'IoU Score = (nan|[\d.]+)', line).group(1)
            dice_score = re.search(r'Dice Score = (nan|[\d.]+)', line).group(1)
            precision = re.search(r'Precision = (nan|[\d.]+)', line).group(1)
            recall = re.search(r'Recall = (nan|[\d.]+)', line).group(1)
            specificity = re.search(r'Specificity = (nan|[\d.]+)', line).group(1)

            val_loss = float(val_loss) if val_loss != 'nan' else float('nan')
            iou_score = float(iou_score) if iou_score != 'nan' else float('nan')
            dice_score = float(dice_score) if dice_score != 'nan' else float('nan')
            precision = float(precision) if precision != 'nan' else float('nan')
            recall = float(recall) if recall != 'nan' else float('nan')
            specificity = float(specificity) if specificity != 'nan' else float('nan')

            data.append([epoch, val_loss, iou_score, dice_score, precision, recall, specificity])

    # Create the 'csv_metrics' folder if it doesn't exist
    os.makedirs('csv_metrics', exist_ok=True)

    # Write .csv file in the 'csv_metrics' folder
    output_csv_path = os.path.join('csv_metrics', output_csv)
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Validation Loss', 'IoU Score', 'Dice Score', 'Precision', 'Recall', 'Specificity'])
        csv_writer.writerows(data)

# This part assumes that you have already unzipped the zip files.
txt_to_csv('output_txt/output_1to5_unlabeled.txt', 'metrics_1to5_unlabeled.csv')
txt_to_csv('output_txt/output_1to5_labeled.txt', 'metrics_1to5_labeled.csv')
txt_to_csv('output_txt/output_1to10_unlabeled.txt', 'metrics_1to10_unlabeled.csv')
txt_to_csv('output_txt/output_1to10_labeled.txt', 'metrics_1to10_labeled.csv')
txt_to_csv('output_txt/output_1to3_unlabeled.txt', 'metrics_1to3_unlabeled.csv')