a general image Crawler to crawl some pictures (

# Crystal Dataset Crawler

This project is a Python script for crawling and processing images from the web.
image rating is based on [DeepGHS/imgutils](https://github.com/deepghs/imgutils)

## Features

- Crawls images from the web based on provided URLs.
- Processes and classifies images based on various scores such as `anime_real_score`, `aesthetic_score`, and `monochrome_score`.
- Deduplicates images based on a similarity threshold.
- Moves images to different folders based on their classification.

## Requirements

The required Python libraries are listed in the requirements.txt file. You can install them using pip:

```sh
pip install -r requirements.txt
```
# Used library
Crawler use [gallery-dl](https://github.com/mikf/gallery-dl) to crawl picture,etc.
Rating image use [imgutils](https://github.com/deepghs/imgutils) to Rate and categories pictures.

# How to use?
Firstly,configure **gallery-dl.conf** at /gallery-dl/gallery-dl.conf to use gallery-dl

write a the links to **./web.txt**，then run the scripts. 

```bash
python main.py --real_threshold VALUE --aesthetic_threshold VALUE --monochrome_threshold VALUE --similarity_threshold VALUE
```   

**Parser info** 

--real_threshold  The threshold for detecting whether an image is a real picture, defaults to 0.9.   
--aesthetic_threshold  The aesthetic score of the image, defaults to 0.35.  
--monochrome_threshold The threshold for detecting whether an image is monochrome, defaults to 0.9  
--similarity_threshold The threshold for performing Phash deduplication on images defaults to 0.2  


**Examples:**  
##SPILTED@taskname1  
website1  
website2  
website3  
##SPILTED@taskname2  
website4  
website5  
website6  
##SPILTED@taskname3  
website7  
website8  
website9  

you will get the result on **.output** 


Crystal 2024



# Change History
7/31 Twitter Crawler is limited to 250 pic per link
