import csv
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_file):
    # Read .csv file
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)  # Skip header row

        data = []
        for row in csv_reader:
            data.append([float(x) for x in row])

    # Transpose data for easier plotting
    data = list(zip(*data))
    
    # Create a 2*3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    
    # Get figure title from the csv file name
    base_name = os.path.basename(csv_file)
    fig_name = os.path.splitext(base_name)[0]
    
    # Add an overall figure title
    fig.suptitle(fig_name, fontsize=16)
    axes = axes.ravel()  # Flatten axes for easier iteration
    
    # Create a plot for each column
    for i in range(1, len(headers)):
        ax = axes[i-1]
        ax.plot(data[0], data[i], marker='o', linestyle='-')
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[i])
        ax.set_title(f'{headers[i]} vs {headers[0]}')
        ax.grid(True)

    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Save the figure as a .png file with the same name as the .csv file, in the 'results_img' folder
    png_file = os.path.join('results_img', fig_name + '.png')
    plt.savefig(png_file)

plot_metrics('csv_metrics/metrics_1to3_unlabeled.csv')
plot_metrics('csv_metrics/metrics_1to5_unlabeled.csv')
plot_metrics('csv_metrics/metrics_1to5_labeled.csv')
plot_metrics('csv_metrics/metrics_1to10_unlabeled.csv')
plot_metrics('csv_metrics/metrics_1to10_labeled.csv')