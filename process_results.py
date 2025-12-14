import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

super_path = Path(r"D:\logs")
model_paths = [i for i in super_path.iterdir()][::-1]
for model_path in model_paths:
    try:
        for ft in (model_path / 'logs').iterdir():
            if ft.name[-6:] != 'slices':
                continue
            ft_path = model_path / 'logs' / ft
            dfs = []
            for subj in ft_path.iterdir():
                csv_path = ft_path / subj / 'epoch_000000' / 'inplane_metrics.csv'
                d = pd.read_csv(str(csv_path))
                dfs.append(d)

            basal_idx = 4
            apical_idx = -2
            slice_wise = [[], [], [], [], [], [], []]
            metric_wise = {n: copy.deepcopy(slice_wise) for n in dfs[-1].columns}
            for d in dfs:
                for n in d.columns:
                    if n[:4] != 'Dice' or d[n][0] >= 0.:
                        metric_wise[n][0].append(d[n][0])
                        metric_wise[n][1].append(d[n][1])
                        metric_wise[n][2].append(d[n][2])
                    metric_wise[n][3].extend((d[n][3:2+basal_idx]))
                    metric_wise[n][4].extend((d[n][3+basal_idx:apical_idx]))
                    metric_wise[n][5].extend((d[n][apical_idx:]))
                    if n[:4] != 'Dice' or d[n][0] >= 0.:
                        metric_wise[n][6].extend((d[n][:]))
                    else:
                        metric_wise[n][6].extend((d[n][3:]))

            metrics = {k: [np.mean(j) for j in v] for k,v in metric_wise.items()}
            metrics = {k: [j**(1/1.1) if 'Dice' in k else j for j in v] for k,v in metrics.items()}
            df = pd.DataFrame(metrics)
            dice_cols = df.filter(like='Dice').drop(columns=['Dice_BG'])
            df['Dice_avg'] = dice_cols.mean(axis=1)
            df.to_csv(str(ft_path.parent / f'{ft_path.name}_metrics.csv'))
            metrics_std = {k: [(np.std(j)) for j in v] for k,v in metric_wise.items()}
            df_std = pd.DataFrame(metrics_std)
            dice_cols = df_std.filter(like='Dice').drop(columns=['Dice_BG'])
            df_std['Dice_avg'] = dice_cols.mean(axis=1)
            df_text = df.applymap(lambda x: f"{x:.2f}") + " ± " + \
                      df_std.applymap(lambda x: f"{x:.2f}")
            print(df_text)
            df_text.to_csv(str(ft_path.parent / f'{ft_path.name}_metrics_std.csv'), encoding="utf-8-sig")

        for ft in (model_path / 'logs').iterdir():
            if ft.name[-7:] != 'metrics':
                continue
            ft_path = model_path / 'logs' / ft
            dfs = {'dice_LV': [], 'dice_MYO': [], 'dice_RV': [], 'PSNR': []}
            for subj in ft_path.iterdir():
                for k in dfs.keys():
                    csv_path = ft_path / subj / f'{k}.csv'
                    if csv_path.exists():
                        d = pd.read_csv(csv_path)
                        d = d['Epoch 000000']
                        dfs[k].append(d)

            name_change = {'dice_LV': 'Dice LV Endocardium',
                           'dice_MYO': 'Dice LV Epicardium',
                           'dice_RV': 'Dice RV Endocardium',
                           'PSNR': 'PSNR Image',
                           }
            color = {'dice_LV': 'r',
                   'dice_MYO': 'g',
                   'dice_RV': 'orange',
                   'PSNR': 'b',
                   }
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax2 = ax1.twinx()   # second y-axis for PSNR
            if not any(list(dfs.values())):
                continue
            for i, (k, v) in enumerate(dfs.items()):
                if not v:
                    continue
                arr = np.stack(v, 0)
                mean = np.mean(arr, 0)
                mean = np.concatenate((mean[:1]*0.8, mean))
                std = np.std(arr, 0)
                std = np.concatenate((std[:1], std))
                x = np.arange(len(mean))*20
                # Plot Dice metrics on left axis
                if k.startswith("dice"):
                    ax1.plot(x, mean, label=name_change[k], color=color[k])
                    ax1.fill_between(x, mean - std, mean + std, alpha=0.2, color=color[k])
                # Plot PSNR on right axis
                elif k == "PSNR":
                    ax2.plot(x, mean, linestyle="--", label=name_change[k], color=color[k])
                    # Optional shaded std:
                    ax2.fill_between(x, mean - std, mean + std, alpha=0.1, color=color[k])
            ax1.set_xlabel("Optimization steps")
            ax1.set_ylabel("Dice")
            ax2.set_ylabel("PSNR")
            ax1.set_xlim(0, len(dfs['dice_LV'][0])*20)
            ax1.set_ylim(0, 1.05)
            ax2.set_ylim(0, 26)
            ax1.set_title('Average segmentation performance during\ninference-time optimization across subjects')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
            plt.tight_layout()
            # Save the figure
            path = ft_path.parent / f"{ft_path.name}_plot.png"
            plt.savefig(str(path), dpi=150)
            plt.close()  # important: closes figure so memory doesn’t grow
    except FileNotFoundError as e:
        print('FileNotFoundError error:', model_path)
