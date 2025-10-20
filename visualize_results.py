import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Directory containing the results
results_dir = "experiments/360_covis_aug_partialscheduler_full"

# Scene names
scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]

# Gather all results
results = {}
for scene in scenes:
    result_file = os.path.join(results_dir, scene, "results.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            if "ours_30000" in data:
                results[scene] = data["ours_30000"]
            else:
                print(f"Warning: No 'ours_30000' key in {scene}")
    else:
        print(f"Warning: {result_file} not found")

# Print summary table
print("\n" + "="*80)
print(f"{'Scene':<15} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12}")
print("="*80)

psnr_values = []
ssim_values = []
lpips_values = []
scene_names = []

for scene in scenes:
    if scene in results:
        r = results[scene]
        psnr = r.get("PSNR", 0)
        ssim = r.get("SSIM", 0)
        lpips = r.get("LPIPS", 0)
        
        print(f"{scene:<15} {psnr:<12.4f} {ssim:<12.4f} {lpips:<12.4f}")
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips)
        scene_names.append(scene)

# Calculate averages
if len(psnr_values) > 0:
    print("="*80)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    print(f"{'Average':<15} {avg_psnr:<12.4f} {avg_ssim:<12.4f} {avg_lpips:<12.4f}")
    print("="*80 + "\n")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('360_covis_aug_partialscheduler_full Results', fontsize=16, fontweight='bold')

# PSNR bar chart
ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(scene_names)), psnr_values, color='steelblue', alpha=0.8)
ax1.axhline(y=avg_psnr, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_psnr:.2f}')
ax1.set_xlabel('Scene', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR by Scene (Higher is Better)', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(scene_names)))
ax1.set_xticklabels(scene_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(psnr_values):
    ax1.text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# SSIM bar chart
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(scene_names)), ssim_values, color='seagreen', alpha=0.8)
ax2.axhline(y=avg_ssim, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_ssim:.4f}')
ax2.set_xlabel('Scene', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM by Scene (Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(scene_names)))
ax2.set_xticklabels(scene_names, rotation=45, ha='right')
ax2.set_ylim([0.7, 1.0])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(ssim_values):
    ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# LPIPS bar chart
ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(scene_names)), lpips_values, color='coral', alpha=0.8)
ax3.axhline(y=avg_lpips, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_lpips:.4f}')
ax3.set_xlabel('Scene', fontsize=12)
ax3.set_ylabel('LPIPS', fontsize=12)
ax3.set_title('LPIPS by Scene (Lower is Better)', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(scene_names)))
ax3.set_xticklabels(scene_names, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(lpips_values):
    ax3.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Combined normalized metrics (for comparison)
ax4 = axes[1, 1]
x = np.arange(len(scene_names))
width = 0.25

# Normalize metrics for visualization (PSNR/40, SSIM*1, LPIPS*5 for scale)
psnr_norm = [p/40 for p in psnr_values]
ssim_norm = ssim_values
lpips_norm = [l*5 for l in lpips_values]  # Scale up for visibility

bars_psnr = ax4.bar(x - width, psnr_norm, width, label='PSNR/40', color='steelblue', alpha=0.8)
bars_ssim = ax4.bar(x, ssim_norm, width, label='SSIM', color='seagreen', alpha=0.8)
bars_lpips = ax4.bar(x + width, lpips_norm, width, label='LPIPS×5', color='coral', alpha=0.8)

ax4.set_xlabel('Scene', fontsize=12)
ax4.set_ylabel('Normalized Value', fontsize=12)
ax4.set_title('Normalized Metrics Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(scene_names, rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('360_covis_aug_partialscheduler_full_results.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved to: 360_covis_aug_partialscheduler_full_results.png")
plt.show()
