# Summarizing News via Telegram Bot
A telegram bot that is able to extract the summary of news from Singapore's Straits Times. 
![Bot in Action](https://github.com/jaotheboss/NewsSummary-Telegram-Bot/blob/master/images/Bot%20in%20Action.png)

## Objective:
To reduce the amount of lines to read the news. To do this, we only extract the essential and relevant sentences to the news.

## Methodology:
Using extractive summarisation, we are able to rank the sentences in the article by cross referencing their relevance with each other. We then use this data to extract the essential and relevant sentences before churning them out as summaries for the news article. To understand extractive summarisation, I'd recommend taking a look at my [Extractive Summariser repo](https://github.com/jaotheboss/Extractive-Summariser)

## Changes Made:
After discovering the use of transformers, i embarked on a journey to use BERT encoding architecture to embed sentences into vectors. Using transformer encoders, i was able to generate vectors that took into account the context of the word in that particular sentence. 

The new sentence ranking algorithm takes takes advantage of vectors that represent more contextual meaning of each sentence compared to simply a pattern of word vectors together. 

![with BERT](https://github.com/jaotheboss/NewsSummary-Telegram-Bot/blob/master/images/Bot%20in%20Action%20using%20BERT.png)

## Notes:
- Key kept in person GDrive
