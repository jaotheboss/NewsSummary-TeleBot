"""
Reference: https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot
"""

import os
os.chdir('/Users/jaoming/Documents/Active Projects/Text Summarisation')

import ExtractiveSummary_forbot as es                   # Extractive Summary module

import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler          # for handling commands ie. /start

# initialising the bot token and the bot itself
bot_token = "1025831290:AAFf5_na0fi7xxHRfB4qNII1mOFe-N2P09k"
bot = telegram.Bot(token = bot_token)
# print(bot.get_me()) # getting details of the bot

# establishing an updater and a dispatcher
"""
Updater class continuously fetches new updates from telegram and passes them on to the Dispatcher class
"""
updater = Updater(token = bot_token, use_context = True)
dispatcher = updater.dispatcher

# setting up the logging module for debugging
import logging
logging.basicConfig(format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                     level = logging.INFO)

# defining functions to process specific types of updates
def start(update, context):
       """
       Function:     To handle the /start update
       """
       context.bot.send_message(chat_id = update.effective_chat.id,
                            text = "Hello! I'm a bot that summarises news based on news links!\n\nHere's how you send me a link!:\n/summarise https://wtv_news_link")

def end(update, context):
       context.bot.send_message(chat_id = update.effective_chat.id,
                            text = "It was good chat. Bye!")

def caps(update, context):
       text_caps = ' '.join(context.args).upper()
       context.bot.send_message(chat_id = update.effective_chat.id,
                            text = text_caps)

def legit_link(update, context):
       link = context.args[0]
       if link[:4] == 'http':
              context.bot.send_message(chat_id = update.effective_chat.id,
                                   text = "You just sent me a link!")
       else:
              context.bot.send_message(chat_id = update.effective_chat.id,
                                   text = "Sorry, this isn't a link!")

def summarise(update, context):
       link = context.args[0]
       summary = es.main(link)
       context.bot.send_message(chat_id = update.effective_chat.id,
                                   text = summary,
                                   parse_mode = telegram.ParseMode.HTML)

def supported_sites(update, context):
       sites = "Straits Times\n"
       context.bot.send_message(chat_id = update.effective_chat.id,
                                   text = "<b>Supported Sites are:</b>\n" + sites,
                                   parse_mode = telegram.ParseMode.HTML)

# creating handlers to handle requests/updates
"""
handle = CommandHandler(name of the button, what will happen when the button is clicked)
dispatcher.add_handler(handle) # add this action to the bot
"""
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

end_handler = CommandHandler('end', end) 
dispatcher.add_handler(end_handler)

caps_handler = CommandHandler('caps', caps)
dispatcher.add_handler(caps_handler)

link_handler = CommandHandler('checkLink', legit_link)
dispatcher.add_handler(link_handler)

summary_handler = CommandHandler('summarise', summarise)
dispatcher.add_handler(summary_handler)

support_handler = CommandHandler('supportedSites', supported_sites)
dispatcher.add_handler(support_handler)

# creating resonses for inline mode
from telegram import InlineQueryResultArticle, InputTextMessageContent
def inline_summarise(update, context):
       query = update.inline_query.query                                     # not working. somehow can't recognize links
       if not query:
              return
       else:
              # summary = es.main(query)
              results = []
              results.append(
                     InlineQueryResultArticle(
                            id = query,
                            title = 'Summarise',
                            input_message_content = InputTextMessageContent(query)
                     )
              )
              context.bot.answer_inline_query(update.inline_query.id, results)

from telegram.ext import InlineQueryHandler
inline_summarise_handler = InlineQueryHandler(inline_summarise)
dispatcher.add_handler(inline_summarise_handler)

# start the bot
updater.start_polling()
