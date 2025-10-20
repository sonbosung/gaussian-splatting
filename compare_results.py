#!/usr/bin/env python3
"""Compare results between original 3DGS and augmented method."""

import json
import os
from pathlib import Path

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
