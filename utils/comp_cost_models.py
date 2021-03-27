import argparse
import os
from ptflops import get_model_complexity_info
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import get_ViT_model, get_resnet_model

parser = argparse.ArgumentParser()

opt = parser.parse_args()
print(opt)

output_path = "../data/"

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
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Plot number of parameters...")
    output_cost_csv = os.path.join(output_path, "models_cost.csv")
    out_df = pd.DataFrame(columns=['model', 'MACs(G)', '#parameters'])

    vit_models = ["ViT-XS", "ViT-S"]
    patch_size = [16, 32]
    resnet_models = ["resnet18"]
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
    for vit in vit_models:
            hybrid_model = get_ViT_model(type=vit, image_size=224, patch_size=16, n_channels=3, n_classes=10,
                                         dropout=0., hybrid=True)
            macs, params = get_model_complexity_info(hybrid_model, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            data_to_add = ["resnet18+"+vit,  str(macs), '{:<8}'.format(params)]
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