import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os
# This script compares the classification agreement between GPT-4o and Gemini models
# for text and image problem size classification.

input_dir = os.path.join(os.getcwd(),"StatsResults")

text_list = [
   os.path.join(input_dir,"PE_Stats_summary_bummer_combined_cgpt_classify_text.csv"),
   os.path.join(input_dir,"PE_Stats_summary_bummer_combined_gemini_classify_text.csv"),
   os.path.join(input_dir,"PE_Stats_summary_disaster_combined_cgpt_classify_text.csv"),
   os.path.join(input_dir,"PE_Stats_summary_disaster_combined_gemini_classify_text.csv"),
   os.path.join(input_dir,"PE_Stats_summary_glitch_combined_cgpt_classify_text.csv"),
   os.path.join(input_dir,"PE_Stats_summary_glitch_combined_gemini_classify_text.csv"),
]
image_list = [
   os.path.join(input_dir,"PE_Stats_summary_bummer_combined_cgpt_classify_image.csv"),
   os.path.join(input_dir,"PE_Stats_summary_bummer_combined_gemini_classify_image.csv"),
   os.path.join(input_dir,"PE_Stats_summary_disaster_combined_cgpt_classify_image.csv"),
   os.path.join(input_dir,"PE_Stats_summary_disaster_combined_gemini_classify_image.csv"),
   os.path.join(input_dir,"PE_Stats_summary_glitch_combined_cgpt_classify_image.csv"),
   os.path.join(input_dir,"PE_Stats_summary_glitch_combined_gemini_classify_image.csv"),
]

# Load and separate into different classification models
gpt4o_dfs = []
gemini_dfs = []
for filetype in ['text','image']:
    if filetype=='text':
        filelist= text_list
    else:
        filelist= image_list        
    for file in filelist:
        df = pd.read_csv(file)
        # Decide which model this file is for based on filename
        if "cgpt" in file.lower() or "gpt4o" in file.lower():
            gpt4o_dfs.append(df[["Predicted Problem Size"]])
        elif "gemini" in file.lower():
            gemini_dfs.append(df[["Predicted Problem Size"]])

    gpt4o_all = pd.concat(gpt4o_dfs, ignore_index=True)
    gemini_all = pd.concat(gemini_dfs, ignore_index=True)
    gpt4o_all["Predicted Problem Size"] = gpt4o_all["Predicted Problem Size"].str.lower()
    gemini_all["Predicted Problem Size"] = gemini_all["Predicted Problem Size"].str.lower()
    min_len = min(len(gpt4o_all), len(gemini_all))
    gpt4o_preds = gpt4o_all["Predicted Problem Size"][:min_len]
    gemini_preds = gemini_all["Predicted Problem Size"][:min_len]


    # Compute Cohen’s kappa 
    kappa = cohen_kappa_score(gpt4o_preds, gemini_preds)

    print(f"Cohen’s kappa (Gpt4o vs Gemini) for {filetype}: {kappa:.3f}\n")