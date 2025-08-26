# this script generates a sets of social scenarios with text, image, and video using OpenAI's GPT-4o and DallE3 or GPTimage
# It first generates a script, and then images, voiceover, and eventually a short video
# Change the problem sizes (glitch, bummer, disaster) to generate 100 of each type of scenario (300 scenarios in total)
# The generated image and video are saved in a specified folder to be analyzed later
# The script also saves the statistics of the generation text in a CSV file

import json
import os
import tempfile
from time import time
import pandas as pd
import random
import base64

import requests
from dotenv import load_dotenv
from moviepy import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)

from openai import OpenAI
#load all the API key from the env file
load_dotenv()

os.environ["OPIK_PROJECT_NAME"] = "video-generation"

PROMPT = """
You are an automated system that helps generate 8-second videos. The user will provide a
prompt, based on which, you will return a script with 5 sentences which meet openAI's content policy. Each sentence of the script will be an
object in the array. 

The object will have the following attributes:

* text - the sentence of the script. 

* image - a prompt that can be sent to GPTimage to generate a
cartoon image for the given sentence that also aligns with the overall context
of the video; the image should have no text in it. 

* voice - A voice url
"""
# 

def generate_script(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": prompt +"limit the script to 4 sentenses",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "script",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scenes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "image": {"type": "string"},
                                    "voice": {"type": "string"},
                                },
                                "required": ["text", "image", "voice"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["scenes"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    return json.loads(response.output_text)

########## This is the DallE-3 image generation ###########
def generate_image(client: OpenAI, prompt: str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        #quality='hd',
        #style="natural",
        n=1,
    )

    return response.data[0].url
########## This is the GPTimage image generation ###########
def generate_image_GPTimage(client:OpenAI, prompt:str)->str:

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
        #quality="high",
        n=1,
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    path= os.path.join(os.getcwd(), "GPTimage.png")
    with open(path, "wb") as f:
        f.write(image_bytes)
        f.close()
    return path


#this is the LemonFox text to voice.
def generate_voiceover_LF(voice:str,text:str,i)->str:
    url = "https://api.lemonfox.ai/v1/audio/speech"
    headers = {
        "Authorization": os.getenv("Lemonfox_API_Key"),
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "voice": voice,
        "response_format": "wav"
    }

    response = requests.post(url, headers=headers, json=data)
  
   #LemonFox can't pass the url of response so the audio has been downloaded 
    output_file = os.path.join(os.getcwd(), f"speech"+str(i)+".wav")

    with open(output_file, "wb") as f:
        
        f.write(response.content)
        f.close
    return output_file



def download_file(url: str, path: str) -> None:
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)

def copy_file(src: str, dest: str) -> None:

    # importing the shutil module
    import shutil
    # create duplicate of the file at the destination with the name mentioned at the end of the destination path
    path = shutil.copyfile(src,dest)


def generate_video(movie: list[dict],i,p,tool,url) -> str:
    clips = []
    for index, scene in enumerate(movie):
        image_url=url
        #LemonFox can't pass the url of response so the audio has been downloaded earlier and can be used directly here
        voiceover_url = scene["voiceover"]
        image_path = os.path.join(os.getcwd(), f"scene_{index}.png")   
        if tool=="DallE3":
        ########code below is for DALLE-3
            # Download the image from the URL
            download_file(image_url, image_path)
        else: #code is different for GPT4oimage due to different API response
            image_path=image_url

        audio_clip = AudioFileClip(voiceover_url)
        video_clip = ImageClip(image_path, duration=(int(audio_clip.duration) + 1))
        video_clip = video_clip.with_audio(audio_clip)
        clips.append(video_clip)

    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(os.path.join(os.getcwd(),f"./{p}Folder",f"video_{p}_{i}_{tool}.mp4"), fps=24, codec="libx264")
    final_video.close()

def add_row(filename,newlist):
    from csv import writer
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(filename, 'a') as f_object:

        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(newlist)

        # Close the file object
        f_object.close()

def main():
    #the problem size can be changed to disaster, bummer,or glitch. Each is run separately due to long processing time and unstability of DALLE3
    problem_size="bummer" #.capitalize()
    stats_columns=["scenario","Image_Tool","Total_Time","Time_Script","Time_Image","Time_Voice","Time_Video","Problem Size", "setting","Script"]
    print(os.getcwd())
    #make folder

    outputfolder=os.path.join(os.getcwd(),f"{problem_size.capitalize()}Folder")
    os.makedirs(outputfolder, exist_ok=True) 
    #add_row(os.path.join(outputfolder,f"Stats_summary_{problem_size}_combined.csv"),stats_columns)


    #automatically read key from the env file
    client = OpenAI()
    #specifiy the number of scenarios to generate
    n=100
    #list of settings used to diversify the settings of the stories; if not used, GPT generates many duplicate scenarios on similar settings.
    setting_list=['volleyball', 'soccer','running', 'basketball','class', 'curling', 'lacrosse', 'singing', 'dancing', 'art', 'after school club', 'birthday party','tryout', 'game', 'field trip', 'swimming','ski','tennis','playing video game','vacation']

    for S_index in range(1,n+1):
        #add timer
        start_time=time()
        #shuffle list
        setting=setting_list[(S_index +random.randint(0,30)) % len(setting_list)]

        print(f'Running {S_index} th scenario')
        #add timer
        before_script=time()
        script = generate_script(
            client,
                prompt = f"""


Tell a short, realistic incident that triggers negative emotions for someone aged 5 to 18 using specific information below. The story will be presented to a child to ask him to identify the size of the problem. Randomly choose their name and gender. Randomly select one setting from the list below. For each problem, describe the problem details, whether there is any quick fix or working back up plan to largely mitigate the issue, and the impact and the duration of the impact. The story ends when the problem presents itself but has not been solved yet and the character asks themselves: "How big is this problem?". Don't tell the size of the problem in the story.

Construct a story related to {setting} to show a problem whose size can be categorized as {problem_size} according to the following definition of each size of problem:
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

Double-check to ensure there is no ambiguity regarding the size of the problem generated. There should be no overlaps between the three categories—if a problem fits the definition of one category, it should not be able to fit the definition of another category.
The generated script must meet OpenAI's content safety policy so it can later be used to create images by DALL-E 3 and GPT-4o.

  """
        )
        after_script=time()
        time_script=after_script-before_script
        #save the total script for output later
        totalscript=''
        for j in range(len(script['scenes'])):
            imagescript=totalscript+script['scenes'][j]["image"]
            totalscript+=script['scenes'][j]["text"]
        print(totalscript)
        special_instruction="In a cartoon style, create a four-panel illustration featuring the same main character consistently across the entire story. No words should be displayed. The incident needs to be clearly visualized. Facial expressions should match the severity of the text/script."
        before_image=time()
         #call GPTimage to generate image based on the same image script and special instruction
        image_url_GPTimage=generate_image_GPTimage(client, imagescript+special_instruction)
        after_image=time()
        #call DALLE3 to generate image based on image script and special instruction
        image_url_DallE3 = generate_image(client, imagescript+special_instruction)
        after_D3_image=time()
        time_GPT_image=after_image-before_image
        time_D3_image=after_D3_image-after_image
        time_voice=0
        i=0
        #combine voice and image to generate video
        for imagetool in ["DallE3","GPTimage"]:
            movie = []
            print(f'working on {imagetool} for scenario {S_index}')
            for scene in script["scenes"]:
                before_voice=time()
                voiceover_url = generate_voiceover_LF("Sarah", scene["text"],i)
                after_voice=time()
                time_voice+=after_voice-before_voice
                i=i+1
                if imagetool=="DallE3":
                    image_url=image_url_DallE3
                else:
                    image_url=image_url_GPTimage
                movie.append(
                    {
                        "image": image_url,
                        "voiceover": voiceover_url,
                    }
                )
            before_video=time()
            #generate the video based on the script and image and save it as mp4 files to be analyzed later
            generate_video(movie,S_index,problem_size,imagetool,image_url)
            after_video=time()
            time_video=after_video-before_video
            total_time=after_video-start_time

            #generate the stats for this scenario and add to csv file

            #print(totalscript)
            if imagetool=="DallE3":
                time_image=time_D3_image
            else:
                time_image=time_GPT_image
           
            newlist=[S_index,imagetool, format(total_time, '.2f'),format(time_script,'.2f'), format(time_image,'.2f'), format(time_voice,'.2f'), format(time_video,'.2f'),problem_size, setting, totalscript]
            #now output stats and lables
            add_row(os.path.join(outputfolder,f"Stats_summary_{problem_size}_combined.csv"),newlist)

            #next we need to save the images to be analyzed later
            
            image_path = os.path.join(outputfolder,f"scenario_{problem_size}_{S_index}_{imagetool}.png")
            if imagetool=="GPTimage":
                copy_file(image_url, image_path)
            else:
                download_file(image_url, image_path)
            


if __name__ == "__main__":
#    for j in range (5): 
        main()
