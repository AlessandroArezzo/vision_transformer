import argparse
import os
from ptflops import get_model_complexity_info
import pandas as pd
import numpy as np

from utils import read_csv_from_path, plot_histo, count_model_parameters, get_ViT_model, get_resnet_model
parser = argparse.ArgumentParser()

parser.add_argument('--graph_type', type=str, default="results", help='type of the graph to plot( results or '
                                                                      'computational_cost')

opt = parser.parse_args()
print(opt)

assert opt.graph_type == "results" or opt.graph_type == "computational_cost", "Test type error"
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
elif opt.graph_type == "computational_cost":
    print("Plot number of parameters...")
    output_cost_csv = os.path.join(output_graph, "models_cost.csv")
    out_df = pd.DataFrame(columns=['model', 'MACs', '#parameters'])

    vit_models = ["ViT-XS", "ViT-S"]
    patch_size = [16, 32]
    #hybrid_models = ["resnet50+ViT-XS", "resnet50+ViT-S"]
    resnet_models = ["resnet18", "resnet34"]
    n_param = []
    for vit in vit_models:
        for patch in patch_size:
            model = get_ViT_model(type=vit, image_size=224, patch_size=patch, n_channels=3, n_classes=10, dropout=0.)
            #n_param.append(count_model_parameters(model))
            macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            data_to_add = [vit+"_"+str(patch), '{:<8}'.format(macs), '{:<8}'.format(params)]
            data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
            out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                   ignore_index=True)
            hybrid_model = get_ViT_model(type=vit, image_size=224, patch_size=patch, n_channels=3, n_classes=10,
                                         dropout=0., hybrid=True)
            macs, params = get_model_complexity_info(hybrid_model, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            data_to_add = ["resnet50+"+vit+"_"+str(patch), '{:<8}'.format(macs), '{:<8}'.format(params)]
            data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
            out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                   ignore_index=True)
    """
    for hybrid in vit_models:
        model = get_ViT_model(type=hybrid, image_size=224, patch_size=16, n_channels=3, n_classes=10, dropout=0.1,
                              hybrid=True)
        n_param.append(count_model_parameters(model))
    """
    for resnet in resnet_models:
        model = get_resnet_model(resnet_type=resnet, n_classes=10)
        #n_param.append(count_model_parameters(model))
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        data_to_add = [resnet, '{:<8}'.format(macs), '{:<8}'.format(params)]
        data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
        out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                               ignore_index=True)
    """
    models = vit_models + hybrid_models + resnet_models
    plot_histo(values=n_param, labels=models, x_label="model", y_label="number of parameters(M)",
               title="Parameters of the models (in milions)", y_lim=[0, 100], path_file=output_param_graph)
    """
    out_df.to_csv(output_cost_csv, index=False, header=True)




