import nltk
import wikipedia
import os

all_wiki_titles = "/home/abhinavj/video_enrichment/"
with open(os.path.join(all_wiki_titles,"wiki_all_titles.txt"),'r') as f:
    titles = f.readlines()



# warning:: Calling `section` on a section that has subheadings will NOT return
#            the full text of all of the subsections. It only gets the text between
#            `section_title` and the next subheading, which is often empty.

with open(os.path.join(all_wiki_titles, "wiki_applications.txt"), 'w') as f:
    for j, page_title in enumerate(titles):
        page_title = page_title.replace('\n', '')
        if j+1>181:
            try:
                print(j + 1, " : ", page_title)
                page = wikipedia.WikipediaPage(title=page_title)
                text = page.section("Applications")
                if text:
                    sentences = nltk.sent_tokenize(text)
                    f.write(sentences[0]+"\n")
                    print("sentence founnd")
            except wikipedia.exceptions.DisambiguationError:
                continue





