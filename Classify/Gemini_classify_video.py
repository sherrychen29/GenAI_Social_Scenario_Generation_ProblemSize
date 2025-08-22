import os
import pandas as pd
import time
import google.generativeai as genai
from dotenv import load_dotenv
# This script classifies the size of a problem from a video using Gemini
# It encodes the video as a base64 string and sends it to the OpenAI API
# The classification is based on a predefined prompt that defines the problem sizes
# Change the problem variable to "glitch", "bummer", or "disaster" as needed
# The script reads videos from a specified directory, classifies them, and saves the

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PROMPT = """
You will view a video telling a short story about a child experiencing a social problem. 
Identify the main problem in the story and classify it into one of three categories based on its size.

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

def classify_video(video_path):
    """Classify the problem size based on the video using Gemini API."""
    try:
        # Upload the video file
        myfile = genai.upload_file(
            path=video_path,
            display_name=os.path.basename(video_path)
        )
        print(f"Uploaded file '{os.path.basename(video_path)}' as: {myfile.uri}")

        # Add a fixed delay to allow the file to process
        print(f"Waiting for the file to process...")
        time.sleep(5)  # Wait for 5 seconds (adjust if necessary)

        # Use the file for classification
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content([
            myfile, PROMPT
        ])
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return "Error"

def main():
    # File paths

    problem= "disaster" #change this to glitch/bummer/disaster as needed
    problem_c= problem.capitalize()
    video_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
    # Load the CSV file containing stories
    input_csv=os.path.join(video_dir,f"Stats_summary_{problem}_combined.csv")
    output_csv=f"PE_Stats_summary_{problem}_combined_gemini_classify_video.csv"

  
    # Read CSV
    df = pd.read_csv(input_csv)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Add columns if missing
    if "Video Path" not in df.columns:
        df["Video Path"] = ""
    if "Predicted Problem Size" not in df.columns:
        df.insert(df.columns.get_loc("Problem Size") + 1, "Predicted Problem Size", "")

    # Loop through rows
    for index, row in df.iterrows():
        tool = row["Image_Tool"]
        scenario = row["scenario"]  # Access the "scenario" column
        video_path = os.path.join(video_dir, f"video_{problem}_{scenario}_{tool}.mp4")
        
        if os.path.exists(video_path):
            try:
                predicted_size = classify_video(video_path)
                df.at[index, "Video Path"] = video_path
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Tool: {tool}, Prediction: {predicted_size}")
            except Exception as e:
                print(f"[Scenario {scenario}] Tool: {tool}, Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Tool: {tool}, Video not found: {video_path}")

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

if __name__ == "__main__":
    main()