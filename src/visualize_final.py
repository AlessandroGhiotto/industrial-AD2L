import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

def export_to_pdf(outputs_list, pdf_path, title_prefix=""):
    """
    outputs_list: list of dicts with {
        'input': PIL Image or np.array,
        'gt': np.array (mask),
        'heatmap': np.array,
        'class_name': str,
        'anomaly_name': str,
        'ap': float
    }
    """
    print(f"Exporting {len(outputs_list)} visualizations to {pdf_path}...")
    with PdfPages(pdf_path) as pdf:
        # Group by class
        by_class = {}
        for out in outputs_list:
            by_class.setdefault(out['class_name'], []).append(out)
            
        for cls in sorted(by_class.keys()):
            items = by_class[cls]
            n = len(items)
            samples_per_page = 4
            
            for i in range(0, n, samples_per_page):
                chunk = items[i : i + samples_per_page]
                n_chunk = len(chunk)
                
                fig, axes = plt.subplots(n_chunk, 3, figsize=(10, 4 * n_chunk),
                                         gridspec_kw={'wspace': 0.05, 'hspace': 0.4})
                if n_chunk == 1:
                    axes = axes[np.newaxis, :]
                
                axes[0, 0].set_title('Input', fontsize=10)
                axes[0, 1].set_title('GT Mask', fontsize=10)
                axes[0, 2].set_title('Heatmap', fontsize=10)
                
                aps = []
                for row, item in enumerate(chunk):
                    # Plot input
                    img = item['input']
                    if isinstance(img, Image.Image):
                        img = np.array(img)
                    axes[row, 0].imshow(img)
                    axes[row, 0].set_ylabel(item['anomaly_name'], fontsize=8, rotation=0, labelpad=70, va='center')
                    
                    # Plot GT
                    axes[row, 1].imshow(item['gt'], cmap='gray')
                    
                    # Plot Heatmap
                    axes[row, 2].imshow(item['heatmap'], cmap='inferno')
                    
                    ap = item.get('ap', float('nan'))
                    ap_text = f"AP={ap:.3f}" if not np.isnan(ap) else 'no GT'
                    axes[row, 2].text(5, 15, ap_text, color='white', fontsize=9,
                                      bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))
                    
                    for ax in axes[row]:
                        ax.axis('off')
                    if not np.isnan(ap):
                        aps.append(ap)
                
                mean_ap_text = f" | Mean AP: {np.mean(aps):.3f}" if aps else ""
                fig.suptitle(f"{title_prefix} {cls}{mean_ap_text}", fontsize=12, y=0.98)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    print("Done.")

def load_mask(path, size=(224, 224)):
    """Helper to load and resize a mask."""
    m = np.array(Image.open(path).convert('L').resize(size, Image.NEAREST))
    return (m > 0).astype(np.float32)


def export_combined_to_pdf(outputs_list, pdf_path, title_prefix="",
                           samples_per_page=3, sort_by=None):
    """Five-column side-by-side PDF: input | GT | PatchCore | U-Net | Ensemble.

    outputs_list: list of dicts with keys:
        'input'        PIL.Image or np.array (the source image)
        'gt'           np.array (HxW), binary mask
        'pc_heatmap'   np.array (HxW), PatchCore map (any range)
        'unet_heatmap' np.array (HxW), U-Net map (any range)
        'ens_heatmap'  np.array (HxW), fused map
        'class_name'   str
        'anomaly_name' str
        'pc_ap'        float, per-image pixel-AP for PatchCore (optional)
        'unet_ap'      float, per-image pixel-AP for U-Net (optional)
        'ens_ap'       float, per-image pixel-AP for ensemble (optional)

    sort_by: optional callable(out_dict) -> sortable key. Use e.g.
        lambda o: o['pc_ap'] - o['unet_ap']  to surface PC-wins first,
        lambda o: o['unet_ap'] - o['pc_ap']  for U-Net-wins.
    """
    print(f"Exporting {len(outputs_list)} comparisons to {pdf_path}...")
    with PdfPages(pdf_path) as pdf:
        by_class = {}
        for out in outputs_list:
            by_class.setdefault(out['class_name'], []).append(out)

        for cls in sorted(by_class.keys()):
            items = by_class[cls]
            if sort_by is not None:
                items = sorted(items, key=sort_by, reverse=True)

            for i in range(0, len(items), samples_per_page):
                chunk = items[i:i + samples_per_page]
                n_chunk = len(chunk)

                fig, axes = plt.subplots(
                    n_chunk, 5,
                    figsize=(14, 3.0 * n_chunk),
                    gridspec_kw={'wspace': 0.04, 'hspace': 0.35},
                )
                if n_chunk == 1:
                    axes = axes[np.newaxis, :]

                col_titles = ['Input', 'GT', 'PatchCore', 'U-Net', 'Ensemble']
                for c, t in enumerate(col_titles):
                    axes[0, c].set_title(t, fontsize=10)

                for row, item in enumerate(chunk):
                    img = item['input']
                    if isinstance(img, Image.Image):
                        img = np.array(img)
                    axes[row, 0].imshow(img)
                    axes[row, 0].set_ylabel(
                        item['anomaly_name'], fontsize=8, rotation=0,
                        labelpad=55, va='center')
                    axes[row, 1].imshow(item['gt'], cmap='gray')

                    for col, key, ap_key in [
                        (2, 'pc_heatmap',   'pc_ap'),
                        (3, 'unet_heatmap', 'unet_ap'),
                        (4, 'ens_heatmap',  'ens_ap'),
                    ]:
                        hm = item[key]
                        axes[row, col].imshow(hm, cmap='inferno',
                                              vmin=float(np.min(hm)),
                                              vmax=float(np.max(hm)))
                        ap = item.get(ap_key, float('nan'))
                        if not np.isnan(ap):
                            axes[row, col].text(
                                5, 15, f"AP={ap:.3f}", color='white',
                                fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.2',
                                          fc='black', alpha=0.7))
                    for ax in axes[row]:
                        ax.set_xticks([]); ax.set_yticks([])

                fig.suptitle(f"{title_prefix} {cls}", fontsize=12, y=0.995)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    print("Done.")
