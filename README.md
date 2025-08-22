# GenAI_Social_Scenario_Generation_ProblemSize
This project is to explore using ChatGPT to generate social story videos for social training
The following packages need to be installed:
base64
requests
dotenv (for API license)
moviepy
openai
google.generativeai
Scikit-learn, Scikit-image
Scipy, statsmodels, pingouin, krippendorff
pandas, numpy, matplotlib

Running steps:
1. Run Scenario Generation (Scenario Generation/Generate_Scenario_text_image_video_PE.py)
    Generated text, image, and videos are saved in DisasterFolder,BummerFolder, and GlitchFolder.
2. Run Classification (Classify/Cgpt_classify_image.py, Cgpt_classify_text.py,Gemini_classify_text.py, Gemini_classify_image.py,Gemini_classify_video.py)
    Results are saved in StatsResults folder
3. Run analysis (confusion matrix,classification power, human image analysis, quantitative image analysis, classification agreement analysis, etc.); 
    Results are saved in StatsResults folder
