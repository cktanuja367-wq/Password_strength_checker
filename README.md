# Password_strength_checker
i created a simple web&lt;&lt; 

# Password Strength Checker

A project I made that takes a password and tells you if it is weak, medium or strong.

## What I did
- Trained a model in Python with scikit-learn on a Kaggle dataset
- Added my own feature that checks for common words and patterns like "password" or "123"
- Tested it on passwords I made up myself and looked at where it got things wrong
- Made a small web page with Streamlit to try it out

## How I built it
I came up with the idea and the extra feature, and I used an AI tool to help me with the code. I then tested it myself, checked where it made mistakes and corrected them.

## Use
It checks a password and says if it is weak, medium or strong, so people can pick a better one before using it.

## Limitations
- The model only learns from the dataset, so it can be wrong on passwords that look different from the ones it was trained on
- The strength labels in the dataset were made by another tool, so my model copies that tool's mistakes
- I tested it on a small set of made-up passwords, so the results are not a full measure of accuracy
- It only judges the password text. It cannot know if a password was already leaked or reused
- Some strong passwords can be displayed as medium, and a password with many words in it can sometimes be displayed as strong

## Future Improvements
I plan to add a cartoon character that shows the password strength. This will make the app more interesting and easier to understand.

## Dataset
I used this Kaggle dataset: https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset

I did not upload it here. To run the project, download it and save it as dataset.csv inside a folder named data.

## How to run
Run these two commands in the terminal:
pip install -r requirements.txt'
streamlit run web_public.py then
The web page opens in your browser.

## Demo Video
If you are interested in seeing how it looks, watch the [demo video](https://drive.google.com/file/d/1L7fKbxql58pvOb7Z0yevJ1q7nPndTLJe/view?usp=drivesdk).

## Files
- password_checker.ipynb: the notebook
- web_public.py: the Streamlit app

## What I learned
- How to train and test a machine learning model with scikit-learn
- How to turn passwords into features a model can use, and how adding my own feature changed the results
- How to look at a model's mistakes (error analysis) instead of only its accuracy
- How to build a simple interface with Streamlit
- I thought building a website was difficult, but I learned how to build one by myself by coding with the help of AI tools
- I learned that with an idea we can build anything

## Author
Tanuja C K
