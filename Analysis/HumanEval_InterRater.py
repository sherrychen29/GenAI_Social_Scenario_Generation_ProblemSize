import pandas as pd
import pingouin as pg
import os
import krippendorff
# interrater reliability analysis of human evaluation ratings
# using ICC(2,1) and Krippendorff's alpha
input_dir = os.path.join(os.getcwd(),"StatsResults")

file=os.path.join(input_dir, "Group Evaluation - combined.csv")
df = pd.read_csv(file)
df['subject']=range(1,601)  # Add a subject column for unique identification
# alignment_subsets = []
# aesthetics_subsets = []
df_alignment = df.loc[:, ['subject','Alignment1', 'Alignment2', 'Alignment3', 'Alignment4']]
df_alignment = df_alignment.apply(pd.to_numeric, errors='coerce')

df_long_alignment = df_alignment.melt(id_vars='subject',
                       var_name='rater',
                       value_name='rating')
print(df_long_alignment.head())

icc_alignment = pg.intraclass_corr(
   data=df_long_alignment,
     targets='subject',
     raters='rater',
     ratings='rating')
icc_alignment.to_csv(os.path.join(os.getcwd(),"StatsResults","icc_alignment.csv"), index=False)
print(icc_alignment)

df_aesthetics = df.loc[:, ['subject','Aesthetics1', 'Aesthetics2', 'Aesthetics3', 'Aesthetics4']]
df_aesthetics = df_aesthetics.apply(pd.to_numeric, errors='coerce')

df_long_aesthetics = df_aesthetics.melt(id_vars='subject',
                       var_name='rater',
                       value_name='rating')
print(df_long_aesthetics.head())

icc_aesthetics = pg.intraclass_corr(
   data=df_long_aesthetics,
     targets='subject',
     raters='rater',
     ratings='rating')
icc_aesthetics.to_csv(os.path.join(os.getcwd(),"StatsResults","icc_aesthetics.csv"), index=False)
print(icc_aesthetics)


# Extract ratings (transpose so raters are rows)
alignment_data = df_alignment[['Alignment1', 'Alignment2', 'Alignment3', 'Alignment4']].to_numpy().T
aesthetics_data = df_aesthetics[['Aesthetics1', 'Aesthetics2', 'Aesthetics3', 'Aesthetics4']].to_numpy().T

# Compute Krippendorff's alpha ()
alpha_alignment = krippendorff.alpha(reliability_data=alignment_data, level_of_measurement='interval')
alpha_aesthetics = krippendorff.alpha(reliability_data=aesthetics_data, level_of_measurement='interval')

print("Krippendorff's Alpha - Alignment:", alpha_alignment)
print("Krippendorff's Alpha - Aesthetics:", alpha_aesthetics)

