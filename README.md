a general image Crawler to crawl some pictures (
# Used library
Crawler use [gallery-dl](https://github.com/mikf/gallery-dl) to crawl picture,etc.
Rating image use [imgutils](https://github.com/deepghs/imgutils) to Rate and categories pictures.

# How to use?
write a the links to **./web.txt**，then run the scripts. 

```bash
python main.py 
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

