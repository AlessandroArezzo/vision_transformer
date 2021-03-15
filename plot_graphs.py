import argparse
import os
from matplotlib import pyplot as plt
import numpy as np

from utils import read_csv_from_path
parser = argparse.ArgumentParser()

parser.add_argument('--graph_type', type=str, default="results", help='type of the graph to plot( results or '
                                                                      'num_parameters')

opt = parser.parse_args()
print(opt)

assert opt.graph_type == "results" or opt.graph_type == "num_parameters", "Test type error"
results_csv_path = "./data/models_results.csv"
output_graph = "./data/graph"
if not os.path.exists(output_graph):
    os.makedirs(output_graph)

if opt.graph_type == "results":
    output_graph = os.path.join(output_graph, "accuracy_compare.png")
    results_dict = read_csv_from_path(file_path=results_csv_path)
    models, acc_models, epochs_models = [], [], []
    for dataset in results_dict.keys():
        output_acc_graph = os.path.join(output_graph, "top_accuracy_compare_"+dataset+".png")
        output_epoch_graph = os.path.join(output_graph, "top_epochs_compare_"+dataset+".png")
        dataset_result = results_dict[dataset]
        for model in dataset_result.keys():
            models.append(model)
            acc_models.append([dataset_result[model]['test_acc']])
            epochs_models.append([dataset_result[model]['epoch']])
        fig, ax = plt.subplots()
        ax.barh(np.arange(len(models)), acc_models, color="blue")
        plt.savefig(output_acc_graph)
        plt.close()
        plt.barh(np.arange(len(models)), epochs_models)
        plt.yticks(models)
        plt.savefig(output_epoch_graph)
        plt.close()





