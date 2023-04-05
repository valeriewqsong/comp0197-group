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
    
    marker_size = 3
    # Create a plot for each column
    for i in range(1, len(headers)):
        ax = axes[i-1]
        ax.plot(data[0], data[i], marker='o', markersize = marker_size, linestyle='-')
        if max(data[i]) < 1:
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim([0, max(data[i])])
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[i])
        ax.set_title(f'{headers[i]} vs {headers[0]}')
        ax.grid(True)

    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Create the val_results folder if it doesn't exist
    os.makedirs('val_results_img', exist_ok=True)
    
    # Save the figure as a .png file with the same name as the .csv file, in the 'val_results_img' folder
    png_file = os.path.join('val_results_img', fig_name + '.png')
    plt.savefig(png_file)

plot_metrics('val_results/fully_sup.csv')
plot_metrics('val_results/1to1_semisup.csv')
plot_metrics('val_results/1to1_sup.csv')
plot_metrics('val_results/1to3_sup.csv')
plot_metrics('val_results/1to3_semisup.csv')
plot_metrics('val_results/1to5_semisup.csv')
plot_metrics('val_results/1to5_sup.csv')
plot_metrics('val_results/1to10_semisup.csv')
plot_metrics('val_results/1to10_sup.csv')


def plot_three_metrics(semisup_csv, sup_csv, fully_sup_csv):
    '''Plot comparison between semi-sup model and its lower bound (trained on the labeled data only) and upper bound (all the data in the dataset are labeled)'''
    # Read data from both CSV files
    headers1, data1 = read_csv_data(semisup_csv)
    headers2, data2 = read_csv_data(sup_csv)
    headers3, data3 = read_csv_data(fully_sup_csv)

    # Transpose data for easier plotting
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    data3 = list(zip(*data3))
    
    # Create a 2*3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    # Function to extract label from CSV file name
    def extract_label(file_name):
        file_name = file_name.replace("val_results/", "")
        file_name = file_name.replace(".csv", "")
        return file_name
    
    marker_size = 3
    # Create a plot for each column
    for i in range(1, len(headers1)):
        ax = axes[i-1]
        # if i == 1:
        #     ax.set_ylim([0, max(data1[i])])
        # elif i == 2:
        #     ax.set_ylim([0.75, 0.81])
        # elif i == 3:
        #     ax.set_ylim([0.84, 0.89])
        # elif i == 4:
        #     ax.set_ylim([0.84, 0.91])
        # elif i == 5:
        #     ax.set_ylim([0.82, 0.91])
        # elif i == 6:
        #     ax.set_ylim([0.8, 0.9])
        ax.plot(data1[0], data1[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(semisup_csv)}')
        ax.plot(data2[0], data2[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(sup_csv)}')
        ax.plot(data3[0], data3[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(fully_sup_csv)}')
        ax.set_xlabel(headers1[0])
        ax.set_ylabel(headers1[i])
        ax.set_title(f'{headers1[i]} vs {headers1[0]}')
        ax.grid(True)
        ax.legend()

    # Get figure title from the first csv file name
    base_name = os.path.basename(semisup_csv)
    fig_name = os.path.splitext(base_name)[0]
    fig_name = fig_name.replace("_semisup", "")
    supertitle = 'comparing_semisup_w_upper_lower_bounds_' + fig_name

    # Add an overall figure title
    fig.suptitle(supertitle, fontsize=16)
    
    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Create the 'val_results_img' folder if it doesn't exist
    os.makedirs('val_results_img', exist_ok=True)
    
    # Save the figure as a .png file with the same name as the first .csv file, in the 'val_results_img' folder
    png_file = os.path.join('val_results_img', supertitle + '.png')
    plt.savefig(png_file)

plot_three_metrics('val_results/1to1_semisup.csv', 'val_results/1to1_sup.csv', 'val_results/fully_sup.csv')    
plot_three_metrics('val_results/1to3_semisup.csv', 'val_results/1to3_sup.csv', 'val_results/fully_sup.csv')    
plot_three_metrics('val_results/1to5_semisup.csv', 'val_results/1to5_sup.csv', 'val_results/fully_sup.csv')
plot_three_metrics('val_results/1to10_semisup.csv', 'val_results/1to10_sup.csv', 'val_results/fully_sup.csv')

def plot_four_metrics(csv_1to1, csv_1to3, csv_1to5, csv_1to10):
    '''Plot comparison between the different ratios of labeled to unlabeled data'''
    # Read data from both CSV files
    headers1, data1 = read_csv_data(csv_1to1)
    headers2, data2 = read_csv_data(csv_1to3)
    headers3, data3 = read_csv_data(csv_1to5)
    headers4, data4 = read_csv_data(csv_1to10)

    # Transpose data for easier plotting
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    data3 = list(zip(*data3))
    data4 = list(zip(*data4))
    
    # Create a 2*3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    # Function to extract label from CSV file name
    def extract_label(file_name):
        file_name = file_name.replace("val_results/", "")
        file_name = file_name.replace(".csv", "")
        return file_name
    
    marker_size = 3
    # Create a plot for each column
    for i in range(1, len(headers1)):
        ax = axes[i-1]
        if max(data4[i]) < 1:
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim([0, max(data4[i])])
        ax.plot(data1[0], data1[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to1)}')
        ax.plot(data2[0], data2[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to3)}')
        ax.plot(data3[0], data3[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to5)}')
        ax.plot(data4[0], data4[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to10)}')
        ax.set_xlabel(headers1[0])
        ax.set_ylabel(headers1[i])
        ax.set_title(f'{headers1[i]} vs {headers1[0]}')
        ax.grid(True)
        ax.legend()

    # Add an overall figure title
    fig.suptitle("Comparison between different ratios of labeled to unlabeled data for semi supervised learning", fontsize=16)
    
    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Create the 'val_results_img' folder if it doesn't exist
    os.makedirs('val_results_img', exist_ok=True)
    
    # Save the figure as a .png file with the same name as the first .csv file, in the 'val_results_img' folder
    png_file = os.path.join('val_results_img', 'comparing_semisup_ratios' + '.png')
    plt.savefig(png_file)
    
plot_four_metrics('val_results/1to1_semisup.csv', 'val_results/1to3_semisup.csv', 'val_results/1to5_semisup.csv', 'val_results/1to10_semisup.csv')

def plot_five_metrics(csv_1to1, csv_1to3, csv_1to5, csv_1to10, csv_fullsup):
    '''Plot comparison between the different ratios of labeled to unlabeled data with the fully supervised model'''
    # Read data from both CSV files
    headers1, data1 = read_csv_data(csv_1to1)
    headers2, data2 = read_csv_data(csv_1to3)
    headers3, data3 = read_csv_data(csv_1to5)
    headers4, data4 = read_csv_data(csv_1to10)
    headers5, data5 = read_csv_data(csv_fullsup)

    # Transpose data for easier plotting
    data1 = list(zip(*data1))
    data2 = list(zip(*data2))
    data3 = list(zip(*data3))
    data4 = list(zip(*data4))
    data5 = list(zip(*data5))
    
    # Create a 2*3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    # Function to extract label from CSV file name
    def extract_label(file_name):
        file_name = file_name.replace("val_results/", "")
        file_name = file_name.replace(".csv", "")
        return file_name
    
    marker_size = 3
    # Create a plot for each column
    for i in range(1, len(headers1)):
        ax = axes[i-1]
        if max(data4[i]) < 1:
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim([0, max(data4[i])])
        ax.plot(data1[0], data1[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to1)}')
        ax.plot(data2[0], data2[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to3)}')
        ax.plot(data3[0], data3[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to5)}')
        ax.plot(data4[0], data4[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_1to10)}')
        ax.plot(data5[0], data5[i], marker='o', markersize = marker_size, linestyle='-', label=f'{extract_label(csv_fullsup)}')
        
        ax.set_xlabel(headers1[0])
        ax.set_ylabel(headers1[i])
        ax.set_title(f'{headers1[i]} vs {headers1[0]}')
        ax.grid(True)
        ax.legend()

    # Add an overall figure title
    fig.suptitle("Comparison between semi supervised learning with different ratios of labeled to unlabeled data and fully supervised learning", fontsize=16)
    
    # Adjust layout for better appearance and provide space for the suptitle
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Create the 'val_results_img' folder if it doesn't exist
    os.makedirs('val_results_img', exist_ok=True)
    
    # Save the figure as a .png file with the same name as the first .csv file, in the 'val_results_img' folder
    png_file = os.path.join('val_results_img', 'comparing_full_semi_sup' + '.png')
    plt.savefig(png_file)
    
plot_five_metrics('val_results/1to1_semisup.csv', 'val_results/1to3_semisup.csv', 'val_results/1to5_semisup.csv', 'val_results/1to10_semisup.csv', 'val_results/fully_sup.csv')