"""
scripts/data_analysis.py
Simple dataset loader and EDA for a folder structured as:
data/
  class1/
    img1.jpg
    ...
  class2/
    ...
If no folder dataset exists, script uses CIFAR-10 samples.
"""
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def analyze_folder(dataset_path='data', max_samples=20, save_fig=False):
    # find subfolders
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not classes:
        print("No class folders found in", dataset_path)
        return

    counts = {}
    samples = []
    for c in classes:
        files = [f for f in os.listdir(os.path.join(dataset_path, c)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        counts[c] = len(files)
        samples.extend([(c, os.path.join(dataset_path, c, files[i])) for i in range(min(len(files), max_samples))])

    print("Class counts:", counts)

    # Plot distribution
    plt.figure(figsize=(8,4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Class distribution")
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.tight_layout()
    if save_fig:
        plt.savefig("class_distribution.png")
    plt.show()

    # show sample images grid
    n = min(8, len(samples))
    plt.figure(figsize=(12,4))
    for i in range(n):
        cls, p = samples[i]
        img = Image.open(p).convert('RGB')
        plt.subplot(1,n,i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
    plt.tight_layout()
    if save_fig:
        plt.savefig("sample_images.png")
    plt.show()

if __name__ == "__main__":
    analyze_folder(dataset_path='../data', save_fig=True)
