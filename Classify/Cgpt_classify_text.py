import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# This script classifies the size of a problem from a script text using GPT-4o
# The classification is based on a predefined prompt that defines the problem sizes
# Change the problem variable to "glitch", "bummer", or "disaster" as needed
# The script reads text from a specified directory, classifies them, and saves the results to a CSV file


# Load all the keys from the .env file
load_dotenv()

PROMPT = """
You will read a short story about a child aged 5 to 18 experiencing a social problem. 
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

def predict_problem_size(client: OpenAI, story: str) -> str:
    """
    Use ChatGPT to classify the size of the problem for a given story.
    """
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": story},
        ],
    )
    return response.output_text.strip().lower()

def main():
    problem= "disaster" #change this to glitch/bummer/disaster as needed
    problem_c= problem #.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c.capitalize()}Folder")
    # Load the CSV file containing stories
    input_file=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_file=f"PE_Stats_summary_{problem}_combined_cgpt_classify_text.csv"
    df = pd.read_csv(input_file)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Verify column names
    print("Columns in the CSV file:", df.columns)

    print("API KEY:", os.getenv("OPENAI_API_KEY"))
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Check if "Predicted Problem Size" column exists
    if "Predicted Problem Size" not in df.columns:
        # Insert the column next to "Problem Size"
        problem_size_index = df.columns.get_loc("Problem Size") + 1
        df.insert(problem_size_index, "Predicted Problem Size", "")  # Create the column if it doesn't exist
    
    # Process every other row
    processed_indices = []
    for index in range(0, len(df), 2):  # Skip every other row
        story = df.at[index, "Script"]  # Access the "Script" column
        scenario = df.at[index, "scenario"]  # Access the "scenario" column
        predicted_size = predict_problem_size(client, story)  # Predict problem size for each story
        print(f"[Scenario {scenario}] Prediction: {predicted_size}")  # Output the prediction with scenario
        df.at[index, "Predicted Problem Size"] = predicted_size  # Override or populate the column
        processed_indices.append(index)  # Track processed rows

    # Remove skipped rows
    df = df.loc[processed_indices].reset_index(drop=True)

    # Remove the "Image_Tool" column
    if "Image_Tool" in df.columns:
        df.drop(columns=["Image_Tool"], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()