import os
import base64
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
# This script classifies the size of a problem from an image using GPT-4o
# It encodes the image as a base64 string and sends it to the OpenAI API
# The classification is based on a predefined prompt that defines the problem sizes
# Change the problem variable to "glitch", "bummer", or "disaster" as needed
# The script reads images from a specified directory, classifies them, and saves the results to a CSV file

# Load all the keys from the .env file 
load_dotenv()

PROMPT = """
You will view an image telling a short story about a child aged 5 to 18 experiencing a social problem. 
Identify the major problem in the story and classify the size of the problem into one of three categories based on the definitions and guideline below.

Problem size guide: 
A disaster is defined as a large-size problem for the child. These problems can pose serious risks to personal health and safety, cause the loss of lives of close friends or family members, or cause large financial loss. These problems typically require significant help from others and take a long time to recover.
Examples of disasters are natural disasters, car accidents, house fires, deaths of close family members, major illnesses where recovery is uncertain, and the loss of life-long savings.
Therefore, if a child is hospitalized due to a major illness, if their safety is at risk, or if a natural disaster—including a fire, flood, or prolonged outage of power, water, or food—occurs, the problem size is a disaster.

A bummer is defined as a medium-size problem for the child. These problems can't be quickly fixed, lack a working backup solution, and have some non-transitory impact. The child needs effort or help from others to solve it over time.
Examples of bummers are major disappointments in competitions, performances, tests, social challenges in long-term relationships, and non-life-threatening illnesses that take time to heal. Suppose a child forgets to bring their favorite pair of goggles to a major swim competition and NO backup goggles are available. In that case, their performance will be severely impacted, and the problem size will fall into the bummer category.
If a child is hospitalized due to a major illness, if their safety is at risk, or if a natural disaster—including a fire, flood, or prolonged outage of power, water, or food—occurs, the problem size is a disaster.

A glitch is defined as a small problem for the child. These problems do not pose a risk to health or safety and can be quickly fixed or with a reasonable backup solution.
Examples of glitches are occasional minor disagreements with friends, a bad hair day, or a small mistake on a test or practice, which will not impact the child's overall performance.
For example, if a child forgets to bring their favorite pair of goggles to a major swim competition, but there IS a backup goggle available, the problem size is a glitch. This is because the child can still perform reasonably while wearing the backup goggles.
If a child is hospitalized due to a major illness, if their safety is at risk, or if a natural disaster—including a fire, flood, or prolonged outage of power, water, or food—occurs, the problem size is a disaster.

There should be no overlap between the three categories, i.e., if a problem fits the definition of one category, it should not fit the definition of another category.
Return only one word — “disaster”, “bummer”, or “glitch” — in lowercase.
Do not include any explanation or extra text/symbols such as quotation marks.
"""

def encode_image(image_path: str) -> str:
    """Encode image as base64 string for OpenAI API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def predict_problem_size(client: OpenAI, image_path: str) -> str:
    """Use GPT-4o to classify the size of the problem from an image"""
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
            ]}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

def main():
    # Paths
    problem= "disaster" #change this to glitch/bummer/disaster as needed
    problem_c= problem.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
 
    #input_file = os.path.join(image_dir, "Stats_summary_bummer_combined_1.csv")
    #output_file = os.path.join(image_dir, "Stats_summary_bummer_combined_cgpt_classify_image_1.csv")
    input_file=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_file=f"PE_Stats_summary_{problem}_combined_cgpt_classify_image.csv"
    df = pd.read_csv(input_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Initialize OpenAI client
    client = OpenAI()

    # Add columns if missing
    if "Image Path" not in df.columns:
        df["Image Path"] = ""
    if "Predicted Problem Size" not in df.columns:
        df.insert(8, "Predicted Problem Size", "")  # Insert column at index 8

    # Filter rows where "Image_Tool" is either "GPTimage" or "DallE3"
    df_filtered = df[df["Image_Tool"].isin(["GPTimage", "DallE3"])]

    # Loop through filtered rows
    for index, row in df_filtered.iterrows():
        tool = row["Image_Tool"]
        scenario = row["scenario"]  # Access the "scenario" column
        image_path = os.path.join(image_dir, f"scenario_{problem}_{scenario}_{tool}.png")
        
        if os.path.exists(image_path):
            try:
                predicted_size = predict_problem_size(client, image_path).lower()
                df.at[index, "Image Path"] = image_path
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Tool: {tool}, Prediction: {predicted_size}")
            except Exception as e:
                print(f"[Scenario {scenario}] Tool: {tool}, Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Tool: {tool}, Image not found: {image_path}")

    # Save result
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()