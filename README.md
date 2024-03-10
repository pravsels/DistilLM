# Distil LM

Understanding things would get a massive boost with better visuals across the board and folks would be open to create a lot more if it was much faster/easier to do so. I believe this generalizes across all domains. 

Creating such visually rich articles can be accelerated by building AI tools such as:

- code completion foundation model (FM) for generating Math animations (using the [Manim](https://github.com/ManimCommunity/manim) library created by [3Blue1Brown](https://www.youtube.com/c/3blue1brown)).

- image generation multi-modal FM for going from sketches to diagrams. editing of generated diagrams with user-created masks and text prompts.


Demo video for manim [workflow](https://www.youtube.com/watch?v=1pGGlP5iRDA). 


## Setting up the environment

Create anaconda environment
```
conda env create -f environment.yml 
```

Activate the environment
```
conda activate dlm
```

## Run UI for Manim 
```
chmod +x *.sh

./run_manim.sh
```
