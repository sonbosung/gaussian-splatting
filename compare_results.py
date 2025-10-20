#!/usr/bin/env python3
"""Compare results between original 3DGS and augmented method."""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Directories to compare
original_dir = Path("experiments/360")
augmented_dir = Path("experiments/360_covis_aug_partialscheduler_full")

# Scenes to compare
scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]

# Metrics to compare
metrics = ["PSNR", "SSIM", "LPIPS"]

print("=" * 100)
print(f"Comparison: Original 3DGS vs Covis Augmented Partial Scheduler")
print("=" * 100)
print()

# Store results for summary
results_original = {}
results_augmented = {}

# Compare each scene
for scene in scenes:
    original_file = original_dir / scene / "results.json"
    augmented_file = augmented_dir / scene / "results.json"
    
    if not original_file.exists():
        print(f"⚠️  {scene}: Original results not found")
        continue
    
    if not augmented_file.exists():
        print(f"⚠️  {scene}: Augmented results not found")
        continue
    
    # Load results
    with open(original_file) as f:
        orig_data = json.load(f)
    
    with open(augmented_file) as f:
        aug_data = json.load(f)
    
    # Get the results (usually under "ours_30000" key)
    orig_results = orig_data.get("ours_30000", {})
    aug_results = aug_data.get("ours_30000", {})
    
    # Store for averaging
    results_original[scene] = orig_results
    results_augmented[scene] = aug_results
    
    # Print comparison
    print(f"Scene: {scene.upper()}")
    print("-" * 100)
    print(f"{'Metric':<10} {'Original':<15} {'Augmented':<15} {'Difference':<15} {'Change':<10}")
    print("-" * 100)
    
    for metric in metrics:
        if metric in orig_results and metric in aug_results:
            orig_val = orig_results[metric]
            aug_val = aug_results[metric]
            diff = aug_val - orig_val
            
            # For LPIPS, lower is better; for PSNR and SSIM, higher is better
            if metric == "LPIPS":
                change = "✓ Better" if diff < 0 else "✗ Worse" if diff > 0 else "= Same"
                pct_change = (diff / orig_val) * 100 if orig_val != 0 else 0
            else:
                change = "✓ Better" if diff > 0 else "✗ Worse" if diff < 0 else "= Same"
                pct_change = (diff / orig_val) * 100 if orig_val != 0 else 0
            
            print(f"{metric:<10} {orig_val:<15.6f} {aug_val:<15.6f} {diff:<+15.6f} {change} ({pct_change:+.2f}%)")
    
    print()

# Calculate averages
print("=" * 100)
print("AVERAGE ACROSS ALL SCENES")
print("=" * 100)
print(f"{'Metric':<10} {'Original':<15} {'Augmented':<15} {'Difference':<15} {'Change':<10}")
print("-" * 100)

for metric in metrics:
    orig_vals = [results_original[s][metric] for s in scenes if s in results_original and metric in results_original[s]]
    aug_vals = [results_augmented[s][metric] for s in scenes if s in results_augmented and metric in results_augmented[s]]
    
    if orig_vals and aug_vals:
        avg_orig = sum(orig_vals) / len(orig_vals)
        avg_aug = sum(aug_vals) / len(aug_vals)
        diff = avg_aug - avg_orig
        
        if metric == "LPIPS":
            change = "✓ Better" if diff < 0 else "✗ Worse" if diff > 0 else "= Same"
            pct_change = (diff / avg_orig) * 100 if avg_orig != 0 else 0
        else:
            change = "✓ Better" if diff > 0 else "✗ Worse" if diff < 0 else "= Same"
            pct_change = (diff / avg_orig) * 100 if avg_orig != 0 else 0
        
        print(f"{metric:<10} {avg_orig:<15.6f} {avg_aug:<15.6f} {diff:<+15.6f} {change} ({pct_change:+.2f}%)")

print()
print("=" * 100)

# Count improvements
print("\nSUMMARY:")
print("-" * 100)
for metric in metrics:
    better = 0
    worse = 0
    same = 0
    
    for scene in scenes:
        if scene in results_original and scene in results_augmented:
            if metric in results_original[scene] and metric in results_augmented[scene]:
                orig_val = results_original[scene][metric]
                aug_val = results_augmented[scene][metric]
                diff = aug_val - orig_val
                
                if metric == "LPIPS":
                    if diff < -1e-6:
                        better += 1
                    elif diff > 1e-6:
                        worse += 1
                    else:
                        same += 1
                else:
                    if diff > 1e-6:
                        better += 1
                    elif diff < -1e-6:
                        worse += 1
                    else:
                        same += 1
    
    total = better + worse + same
    if total > 0:
        print(f"{metric}: {better}/{total} scenes improved, {worse}/{total} scenes worse, {same}/{total} same")

print("=" * 100)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Original 3DGS vs Covis Augmented Partial Scheduler', fontsize=16, fontweight='bold')

# Prepare data
scene_names = [s.capitalize() for s in scenes if s in results_original and s in results_augmented]
scene_indices = np.arange(len(scene_names))

# Extract metric values
psnr_orig = [results_original[s]["PSNR"] for s in scenes if s in results_original]
psnr_aug = [results_augmented[s]["PSNR"] for s in scenes if s in results_augmented]

ssim_orig = [results_original[s]["SSIM"] for s in scenes if s in results_original]
ssim_aug = [results_augmented[s]["SSIM"] for s in scenes if s in results_augmented]

lpips_orig = [results_original[s]["LPIPS"] for s in scenes if s in results_original]
lpips_aug = [results_augmented[s]["LPIPS"] for s in scenes if s in results_augmented]

# Plot 1: PSNR comparison
ax1 = axes[0, 0]
bar_width = 0.35
x = scene_indices
ax1.bar(x - bar_width/2, psnr_orig, bar_width, label='Original 3DGS', color='#4A90E2', alpha=0.8)
ax1.bar(x + bar_width/2, psnr_aug, bar_width, label='Covis Aug', color='#E24A4A', alpha=0.8)
ax1.set_xlabel('Scene', fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontweight='bold')
ax1.set_title('PSNR Comparison (Higher is Better)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scene_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add average line
avg_orig_psnr = sum(psnr_orig) / len(psnr_orig)
avg_aug_psnr = sum(psnr_aug) / len(psnr_aug)
ax1.axhline(y=avg_orig_psnr, color='#4A90E2', linestyle='--', alpha=0.5, linewidth=2)
ax1.axhline(y=avg_aug_psnr, color='#E24A4A', linestyle='--', alpha=0.5, linewidth=2)

# Plot 2: SSIM comparison
ax2 = axes[0, 1]
ax2.bar(x - bar_width/2, ssim_orig, bar_width, label='Original 3DGS', color='#4A90E2', alpha=0.8)
ax2.bar(x + bar_width/2, ssim_aug, bar_width, label='Covis Aug', color='#E24A4A', alpha=0.8)
ax2.set_xlabel('Scene', fontweight='bold')
ax2.set_ylabel('SSIM', fontweight='bold')
ax2.set_title('SSIM Comparison (Higher is Better)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(scene_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add average line
avg_orig_ssim = sum(ssim_orig) / len(ssim_orig)
avg_aug_ssim = sum(ssim_aug) / len(ssim_aug)
ax2.axhline(y=avg_orig_ssim, color='#4A90E2', linestyle='--', alpha=0.5, linewidth=2)
ax2.axhline(y=avg_aug_ssim, color='#E24A4A', linestyle='--', alpha=0.5, linewidth=2)

# Plot 3: LPIPS comparison
ax3 = axes[1, 0]
ax3.bar(x - bar_width/2, lpips_orig, bar_width, label='Original 3DGS', color='#4A90E2', alpha=0.8)
ax3.bar(x + bar_width/2, lpips_aug, bar_width, label='Covis Aug', color='#E24A4A', alpha=0.8)
ax3.set_xlabel('Scene', fontweight='bold')
ax3.set_ylabel('LPIPS', fontweight='bold')
ax3.set_title('LPIPS Comparison (Lower is Better)', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(scene_names, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add average line
avg_orig_lpips = sum(lpips_orig) / len(lpips_orig)
avg_aug_lpips = sum(lpips_aug) / len(lpips_aug)
ax3.axhline(y=avg_orig_lpips, color='#4A90E2', linestyle='--', alpha=0.5, linewidth=2)
ax3.axhline(y=avg_aug_lpips, color='#E24A4A', linestyle='--', alpha=0.5, linewidth=2)

# Plot 4: Improvement percentages
ax4 = axes[1, 1]
psnr_improvements = [(psnr_aug[i] - psnr_orig[i]) / psnr_orig[i] * 100 for i in range(len(psnr_orig))]
ssim_improvements = [(ssim_aug[i] - ssim_orig[i]) / ssim_orig[i] * 100 for i in range(len(ssim_orig))]
lpips_improvements = [(lpips_aug[i] - lpips_orig[i]) / lpips_orig[i] * 100 for i in range(len(lpips_orig))]

bar_width_imp = 0.25
x_imp = scene_indices
ax4.bar(x_imp - bar_width_imp, psnr_improvements, bar_width_imp, label='PSNR', color='#50C878', alpha=0.8)
ax4.bar(x_imp, ssim_improvements, bar_width_imp, label='SSIM', color='#FFB347', alpha=0.8)
ax4.bar(x_imp + bar_width_imp, lpips_improvements, bar_width_imp, label='LPIPS', color='#9370DB', alpha=0.8)
ax4.set_xlabel('Scene', fontweight='bold')
ax4.set_ylabel('Improvement (%)', fontweight='bold')
ax4.set_title('Relative Improvement (Covis Aug vs Original)', fontweight='bold')
ax4.set_xticks(x_imp)
ax4.set_xticklabels(scene_names, rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add text box with summary statistics
summary_text = f"Average Improvements:\n"
summary_text += f"PSNR: {avg_aug_psnr - avg_orig_psnr:+.4f} dB ({(avg_aug_psnr - avg_orig_psnr) / avg_orig_psnr * 100:+.2f}%)\n"
summary_text += f"SSIM: {avg_aug_ssim - avg_orig_ssim:+.6f} ({(avg_aug_ssim - avg_orig_ssim) / avg_orig_ssim * 100:+.2f}%)\n"
summary_text += f"LPIPS: {avg_aug_lpips - avg_orig_lpips:+.6f} ({(avg_aug_lpips - avg_orig_lpips) / avg_orig_lpips * 100:+.2f}%)"

fig.text(0.98, 0.02, summary_text, fontsize=11, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save the figure
output_file = "3dgs_comparison_results.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")
plt.close()
