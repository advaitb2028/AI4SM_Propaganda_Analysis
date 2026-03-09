# Proppy Corpus 1.0

Version 1.0: March 25, 2019
---------------------------

This file describes the corpus Proppy 1.0. This is the corpus used in the 
paper "Proppy: Organizing the News Based on Their Propagandistic Content" 
(see citation section).

The corpus contains 52k articles from 100+ news outlets. Each article is 
labeled as either “propagandistic” (positive class) or “non-propagandistic” 
(negative class). The labeling was done indirectly using a technique known as 
distant supervision, i.e. an article is considered propagandistic if it comes 
from a news outlet that has been labeled as propagandistic by human annotators. 


## Data format

We provide the corpus in three tsv files, including training, development, and 
testing partitions.

The data is tab-separated. Each line represents one article, with the following 
information:

1. article_text: the text of the article retrieved via newspaper3k package.
2. event_location: the geographical location - collected from GDELT.
3. average_tone: measures the impact of the event - collected from GDELT
4. article_date: article's publish date - collected from GDELT.
5. article_ID: GDELT ID , unique among the dataset's articles.
6. article_URL: the direct URL for the published article in its source website.
7. MBFC_factuality_label: factuality label for the source from MBFC
8. article_URL
9. MBFC_factuality_label   
10. URL_to_MBFC_page        
11. source_name     
12. MBFC_notes_about_source
13. MBFC_bias_label 
14. source_URL
15. propaganda_label


## About

The corpus was downloaded using MBFC metadata to identify propagandistic vs 
non-propagandistic sources. Specific URLs where then gathered with GDELT and 
contents downloaded with newspaper3k

## Credit

Please cite the following paper when using this corpus:

A. Barrón-Cedeño, G. Da San Martino, I. Jaradat, and P. Nakov. 
Proppy: Organizing news coverage on the basis of their propagandistic content.
Information Processing and Management 56(5), pp. 1849-1864. 2019

@article{BARRONCEDENO20191849,
author = "Barr\'{o}n-Cede\~no, Alberto and
    Da San Martino, Giovanni and
    Jaradat, Israa and
    Nakov, Preslav",
title = "{Proppy: Organizing the news based on their propagandistic content}",
journal = "Information Processing & Management",
volume = "56",
number = "5",
pages = "1849 - 1864",
year = "2019",
issn = "0306-4573",
doi = "https://doi.org/10.1016/j.ipm.2019.03.005",
url = "http://www.sciencedirect.com/science/article/pii/S0306457318306058",
}

## Authors

Alberto Barrón-Cedeno;
Israa Jaradat;
Giovani Da San Martino;
Preslav Nakov
