import argparse
import os
from matplotlib import pyplot as plt
import numpy as np

from utils import read_csv_from_path, plot_histo, count_model_parameters, get_ViT_model, get_resnet_model
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
    print("Plot results graph...")
    results_dict = read_csv_from_path(file_path=results_csv_path)
    for dataset in results_dict.keys():
        models, acc_models, epochs_models = [], [], []
        output_acc_graph = os.path.join(output_graph, "top_accuracy_compare_"+dataset+".png")
        output_epoch_graph = os.path.join(output_graph, "top_epochs_compare_"+dataset+".png")
        dataset_result = results_dict[dataset]
        for model in dataset_result.keys():
            models.append(model)
            acc_models.append(dataset_result[model]['test_acc'])
            epochs_models.append(dataset_result[model]['epoch'])
        plot_histo(values=acc_models, labels=models, x_label="model", y_label="top-1 accuracy",
                   title="Top-1 accuracy "+dataset, y_lim=[0, 100], path_file=output_acc_graph)
        plot_histo(values=epochs_models, labels=models, x_label="model", y_label="top epoch",
                   title="Top epoch "+dataset, y_lim=[0, 100], path_file=output_epoch_graph)
elif opt.graph_type == "num_parameters":
    print("Plot number of parameters...")
    output_param_graph = os.path.join(output_graph, "n_params_compare.png")
    vit_models = ["ViT-XS", "ViT-S"]
    hybrid_models = ["resnet50+ViT-XS", "resnet50+ViT-S"]
    resnet_models = ["resnet18", "resnet34", "resnet50"]
    n_param = []
    for vit in vit_models:
        model = get_ViT_model(type=vit, image_size=224, patch_size=16, n_channels=3, n_classes=10, dropout=0.1)
        n_param.append(count_model_parameters(model))
    for hybrid in vit_models:
        model = get_ViT_model(type=hybrid, image_size=224, patch_size=16, n_channels=3, n_classes=10, dropout=0.1,
                              hybrid=True)
        n_param.append(count_model_parameters(model))
    for resnet in resnet_models:
        model = get_resnet_model(resnet_type=resnet, n_classes=10)
        n_param.append(count_model_parameters(model))
    models = vit_models + hybrid_models + resnet_models
    plot_histo(values=n_param, labels=models, x_label="model", y_label="number of parameters(M)",
               title="Parameters of the models (in milions)", y_lim=[0, 100], path_file=output_param_graph)



