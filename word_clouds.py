import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt



codeup_pic = np.array(Image.open("images/c.png"))
print(codeup_pic.shape)



def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

def re_format(val):
    if val == 255:
        return 0
    else:
        return val

# Transform your mask into a new one that will work with the function:
transformed_codeup_pic = codeup_pic

for i in range(len(codeup_pic)):
    transformed_codeup_pic[i] = list(map(transform_format, codeup_pic[i]))


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def flatten(two_d_list):
    return [word for sub in two_d_list for word in sub]
    

word = [str(path).split('/') for path in df.path.unique()]
word = flatten(word)
word= [w.split('-') for w in word]
word = flatten(word)
word = [w.split('_') for w in word]
word = flatten(word)
word = [w.split('.') for w in word]
word = flatten(word)
    
text = " ".join(word)

stopwords = ['content', 'favicon', 'intro', 'appendix', 'ico', 'overview', 'md', 'to', 'html css']
# Create and generate a word cloud image:

print(transformed_codeup_pic)
wordcloud = WordCloud(width=1200, background_color= 'white',colormap='Greens', height=1200,stopwords=stopwords, mask=transformed_codeup_pic).generate(text)

for i in range(len(codeup_pic)):
    transformed_codeup_pic[i] = list(map(re_format, codeup_pic[i]))


# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
