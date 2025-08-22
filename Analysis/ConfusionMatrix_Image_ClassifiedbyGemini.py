import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Confusion matrix for problem size classification by Gemini using image data
import os
input_dir = os.path.join(os.getcwd(),"StatsResults")

# List of CSV files to load
csv_files = [
    os.path.join(input_dir, "PE_Stats_summary_bummer_combined_gemini_classify_image.csv"),
    os.path.join(input_dir, "PE_Stats_summary_disaster_combined_gemini_classify_image.csv"),
    os.path.join(input_dir, "PE_Stats_summary_glitch_combined_gemini_classify_image.csv")
]
for imagetype in ['GPTimage', 'DallE3']:
# Load and combine data, filtering rows where "Image_Tool" is "DallE3"
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df = df[df["Image_Tool"] == imagetype]  # Filter rows where "Image_Tool" is the current imagetype
        dfs.append(df[["Problem Size", "Predicted Problem Size"]])

    df_all = pd.concat(dfs, ignore_index=True)

    # Define labels
    labels = ["glitch", "bummer", "disaster"]
    df_all["Problem Size"] = df_all["Problem Size"].str.lower()
    df_all["Predicted Problem Size"] = df_all["Predicted Problem Size"].str.lower()

    # Calculate precision, recall, and F1 score
    for label in labels:
        precision = precision_score(df_all["Problem Size"], df_all["Predicted Problem Size"], average='weighted', labels=[label])
        recall = recall_score(df_all["Problem Size"], df_all["Predicted Problem Size"], average='weighted', labels=[label])
        f1 = f1_score(df_all["Problem Size"], df_all["Predicted Problem Size"], average='weighted', labels=[label])
        print(f"Metrics for {label}:")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1 Score: {f1:.2f}")

     # Compute confusion matrix
    cm = confusion_matrix(df_all["Problem Size"], df_all["Predicted Problem Size"], labels=labels)

        # Normalize by row (true labels) to get percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        # Plot normalized confusion matrix (with numbers, no % sign)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["Glitch", "Bummer", "Disaster"])
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', values_format=".2f", ax=ax)

        # Set color scale limits manually
    im = ax.images[0]  # Access the image object created by ConfusionMatrixDisplay
    im.set_clim(0, 100)  # Set vmin=0 and vmax=100

    plt.title(f"Confusion Matrix of {imagetype} Images Classified by Gemini (%)")
    plt.show()

    # Print matrix with percentage signs
    cm_percentage_with_sign = np.array([[f"{value:.2f}%" for value in row] for row in cm_percentage])
    print(f"Confusion Matrix with Percentage Signs for {imagetype} Images:")
    print(cm_percentage_with_sign)