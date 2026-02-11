import matplotlib.pyplot as plt


class BasePlot:

    def _setup_figure(self, figsize=(8, 6)):
        plt.figure(figsize=figsize)

    def _finalize(self, save_path: str | None):
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
