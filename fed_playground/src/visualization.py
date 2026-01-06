
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

class DivergencePlotter:
    """
    Handles plotting of divergence metrics for federated learning experiments.
    """
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir
        self.results = {}
        
    def add_result(self, x_value: Any, metrics_per_round: List[Dict]):
        """
        Store results for a specific x-axis value (e.g. number of instances).
        """
        self.results[x_value] = metrics_per_round
        
    def _collect_series(self, x_values, metric_key, use_locals):
        series = []
        for x in x_values:
            rounds_list = self.results[x]
            if use_locals:
                vals = []
                for md in rounds_list:
                    vals.extend(md.get(f"local_{metric_key}", []))
                series.append(vals)
            else:
                series.append([md[metric_key] for md in rounds_list])
        return series

    def plot(
        self,
        x_label: str,
        title_suffix: str,
        show_single_models: bool = True
    ):
        x_values = sorted(self.results.keys())
        
        metric_configs = [
            ("normdiff", "Norm Difference", "norm_difference.png"),
            ("msediff", "MSE Difference", "mse_difference.png"),
            ("mseratio", "MSE Ratio", "mse_ratio.png"),
        ]
        
        for key, label, fname in metric_configs:
             self._paired_boxplot(
                 x_values, 
                 key, 
                 label, 
                 f"{label} vs {title_suffix}", 
                 fname,
                 x_label,
                 show_single_models
             )
             
    def _paired_boxplot(
        self, 
        x_values, 
        metric_key, 
        ylabel_left, 
        title, 
        fname, 
        x_label, 
        show_single_models
    ):
        base = np.arange(len(x_values), dtype=float)
        width = 0.35

        agg = self._collect_series(x_values, metric_key, use_locals=False)
        loc = (
            self._collect_series(x_values, metric_key, use_locals=True) if show_single_models else None
        )
        
        # Calculate MSE lines
        agg_mse_vals = [np.mean([r["mse"] for r in self.results[x]]) for x in x_values]
        # General MSE (assume same for all rounds, take first)
        gen_mse_vals = [self.results[x][0]["general_mse"] for x in x_values]

        fig, ax = plt.subplots(figsize=(15, 10))

        # Boxplots
        bp_agg = ax.boxplot(
            agg,
            positions=base - (width / 2 if show_single_models else 0.0),
            widths=width,
            patch_artist=True,
            tick_labels=None,
        )
        for b in bp_agg["boxes"]:
            b.set_alpha(0.7)
            b.set_facecolor('lightblue')

        axis_for_locals = ax
        if show_single_models:
             # Use same axis for boxplots if possible, or maybe just offset?
             # User script used twinx sometimes. Let's keep it simple: same scale usually makes sense for diffs?
             # Actually, local models might have huge variance. Let's stick to user logic: 
             # "if show_single_models and twin_y_axis: axis_for_locals = ax.twinx()"
             # Let's simplify and use same axis for boxplots to be comparable.
             
             bp_loc = ax.boxplot(
                loc,
                positions=base + width / 2,
                widths=width,
                patch_artist=True,
                tick_labels=None,
             )
             for b in bp_loc["boxes"]:
                b.set_alpha(0.4)
                b.set_facecolor('orange')

        ax.set_xticks(base)
        ax.set_xticklabels(x_values)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel_left)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.75)
        
        # Plot MSE Lines on secondary axis
        ax_mse = ax.twinx()
        ax_mse.set_ylabel("MSE")
        
        (line_gen,) = ax_mse.plot(base, gen_mse_vals, color='green', marker='x', linestyle='--', label='General MSE')
        (line_agg,) = ax_mse.plot(base, agg_mse_vals, color='blue', marker='o', linestyle=':', label='Aggregated MSE')
        
        handles = [
            plt.Line2D([0],[0], color='lightblue', lw=4, alpha=0.7, label='Aggregated Box'),
            line_gen,
            line_agg
        ]
        
        if show_single_models:
             loc_mse_vals = []
             for x in x_values:
                 all_l = []
                 for r in self.results[x]:
                     all_l.extend(r["local_mse"])
                 loc_mse_vals.append(np.mean(all_l))
            
             (line_loc,) = ax_mse.plot(base, loc_mse_vals, color='orange', marker='s', linestyle=':', label='Single Models MSE')
             handles.append(plt.Line2D([0],[0], color='orange', lw=4, alpha=0.4, label='Single Models Box'))
             handles.append(line_loc)
             
        ax.legend(handles=handles, loc='upper center', ncol=3)
        
        plt.tight_layout()
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, fname))
            plt.close()
        else:
            plt.show()
