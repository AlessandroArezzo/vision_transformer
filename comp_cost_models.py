import argparse
import csv
import os
from ptflops import get_model_complexity_info
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import get_ViT_model, get_resnet_model, read_csv_from_path

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, default="computational_cost", help='type of the graph to plot( results or '
                                                                      'computational_cost')
opt = parser.parse_args()
print(opt)

assert opt.graph_type == "comp_cost_results" or opt.graph_type == "computational_cost", "Test type error"
output_graph = "./data/graph"

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

if __name__ == '__main__':
    if opt.graph_type == "computational_cost":

        if not os.path.exists(output_graph):
            os.makedirs(output_graph)
        print("Plot number of parameters...")
        output_cost_csv = os.path.join(output_graph, "models_cost.csv")
        out_df = pd.DataFrame(columns=['model', 'MACs(G)', '#parameters'])

        vit_models = ["ViT-XS", "ViT-S"]
        patch_size = [16, 32]
        resnet_models = ["resnet18", "resnet34"]
        n_param = []
        for vit in vit_models:
            for patch in patch_size:
                model = get_ViT_model(type=vit, image_size=224, patch_size=patch, n_channels=3, n_classes=10, dropout=0.)
                macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
                data_to_add = [vit+"_"+str(patch), str(macs), '{:<8}'.format(params)]
                data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
                out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                       ignore_index=True)
                hybrid_model = get_ViT_model(type=vit, image_size=224, patch_size=patch, n_channels=3, n_classes=10,
                                             dropout=0., hybrid=True)
                macs, params = get_model_complexity_info(hybrid_model, (3, 224, 224), as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
                data_to_add = ["resnet18+"+vit+"_"+str(patch),  str(macs), '{:<8}'.format(params)]
                data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
                out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                       ignore_index=True)
        for resnet in resnet_models:
            model = get_resnet_model(resnet_type=resnet, n_classes=10)
            macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            data_to_add = [resnet, str(macs), '{:<8}'.format(params)]
            data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
            out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                   ignore_index=True)
        out_df.to_csv(output_cost_csv, index=False, header=True)

    elif opt.graph_type == "comp_cost_results":
        print("Plot results graph...")
        results_csv_path = "./data/models_results.csv"
        comp_cost_csv_path = os.path.join(output_graph, "models_cost.csv")
        graph_path = "./data/compare_cost_result.png"
        cost = {}
        with open(comp_cost_csv_path, 'r') as data_file:
            reader = csv.reader(data_file)
            for idx, row in enumerate(reader):
                if idx > 0:
                    model = row[0]
                    macs = row[1].split(" ")[0]
                    cost[model] = float(macs)
        results_dict = read_csv_from_path(csv_path=results_csv_path)
        models = {"vit_32": {"models": ["ViT-XS_32", "ViT-S_32"], "marker": "v", "color": "green"},
                  "vit_16": {"models": ["ViT-XS_16", "ViT-S_16"], "marker": "o", "color": "blue"},
                  "hybrid_vit": {"models": ["resnet18+ViT-XS", "resnet18+ViT-S"], "marker": "P", "color": "orange"},
                  "resnet": {"models": ["resnet18"], "marker": "s", "color": "black"}}
        x, y, markers, colors = [], [], [], []
        scatters = []
        for model_type in models.keys():
            for model in models[model_type]["models"]:
                macs = cost[model]
                sum_acc, count = 0, 0
                scatters_model = []
                for dataset in results_dict.keys():
                    try:
                        sum_acc += float(results_dict[dataset][model]['test_acc'])
                        count += 1
                    except KeyError:
                        continue
                if count:
                    mean_acc = sum_acc / count
                    x.append(macs)
                    y.append(mean_acc)
                    markers.append(models[model_type]["marker"])
                    colors.append(models[model_type]["color"])
                    scatter = plt.scatter(macs, mean_acc, marker=models[model_type]["marker"],
                                color=models[model_type]["color"], s=120)
                    scatters_model.append(scatter)
            scatters.append(scatters_model)
        #plt.legend()
        #plt.legend(scatters, ["ViT_32", "ViT_16", "resnet18+ViT", "resnet18"])
        #fig, ax = plt.subplots()
        #scatter = mscatter(x, y, c=colors, m=markers, ax=ax)

        #plt.scatter(x, y, markers=markers, c=colors)
        plt.xlabel("MACs")
        plt.ylabel("accuracy")
        plt.title("Average accuracy as a function of the computational costs")
        plt.savefig(graph_path)