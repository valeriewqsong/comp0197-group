import csv
import matplotlib.pyplot as plt
import os

def read_csv_data(csv_file):
    '''Function to read CSV data'''
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        data = []
        for row in csv_reader:
            data.append([float(x) for x in row])
    return headers, data


def plot_metrics(csv_file):
    '''Plot metrics based on one csv file'''
    # Read .csv file
    headers, data = read_csv_data(csv_file)

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
        ax.set_ylim([0, 1])
        ax.plot(data[0], data[i], marker='o', linestyle='-')
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[i])
        ax.set_title(f'{headers[i]} vs {headers[0]}')
        ax.grid(True)

    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Save the figure as a .png file with the same name as the .csv file, in the 'results_img' folder
    # Note that the results_img folder must already exist
    png_file = os.path.join('results_img', fig_name + '.png')
    plt.savefig(png_file)

plot_metrics('csv_metrics/metrics_1to3_semisup.csv')
plot_metrics('csv_metrics/metrics_1to5_semisup.csv')
plot_metrics('csv_metrics/metrics_1to5_sup.csv')
plot_metrics('csv_metrics/metrics_1to10_semisup.csv')
plot_metrics('csv_metrics/metrics_1to10_sup.csv')


def plot_two_metrics(csv_file1, csv_file2):
    '''Plot comparison between semi-sup model and its lower bound (trained on sup only)'''
    # Read data from both CSV files
    headers1, data1 = read_csv_data(csv_file1)
    headers2, data2 = read_csv_data(csv_file2)

    # Transpose data for easier plotting
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    
    # Create a 2*3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    # Create a plot for each column
    for i in range(1, len(headers1)):
        ax = axes[i-1]
        ax.set_ylim([0, 1])
        ax.plot(data1[0], data1[i], marker='o', linestyle='-', label=f'{csv_file1}')
        ax.plot(data2[0], data2[i], marker='o', linestyle='-', label=f'{csv_file2}')
        ax.set_xlabel(headers1[0])
        ax.set_ylabel(headers1[i])
        ax.set_title(f'{headers1[i]} vs {headers1[0]}')
        ax.grid(True)
        ax.legend()

    # Get figure title from the first csv file name
    base_name = os.path.basename(csv_file1)
    fig_name = os.path.splitext(base_name)[0]
    suptitle = fig_name.split("_", 2)[1]
    supertitle = 'semisup comparison w/ sup ' + suptitle

    # Add an overall figure title
    fig.suptitle(supertitle, fontsize=16)
    
    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Create the 'results_img' folder if it doesn't exist
    os.makedirs('results_img', exist_ok=True)
    
    # Save the figure as a .png file with the same name as the first .csv file, in the 'results_img' folder
    png_file = os.path.join('results_img', 'semisup_comparison_w_sup_' + suptitle + '.png')
    plt.savefig(png_file)
    
plot_two_metrics('csv_metrics/metrics_1to5_semisup.csv', 'csv_metrics/metrics_1to5_sup.csv')
plot_two_metrics('csv_metrics/metrics_1to10_semisup.csv', 'csv_metrics/metrics_1to10_sup.csv')