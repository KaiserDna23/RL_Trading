# Telegram notifier for IA agent

import logging
import os
import requests
from dotenv import load_dotenv

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# define handlers

def getBotAPI(config_file:str):
    load_dotenv(config_file)
    return os.getenv("bot_api")

def getUpdates(token:str):
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    result = requests.get(url).json()
    #print(result)
    return result

def getBotChatID(token:str, check=False):
    rslt = getUpdates(token)
    if isinstance(rslt['result'], dict):
        rslt = dict(rslt['result'][0])
        return rslt["message"]["chat"]["id"]
    else:
        # hard coded
        return '545XXXXXXXX'



def send_message(message_str = None):
    config_file = r"../sec/tel.env"
    api_ = getBotAPI(config_file)
    chat_id = getBotChatID(api_, check=True)
    #print(chat_id)
    req = f"https://api.telegram.org/bot{str(api_)}/sendMessage?chat_id={str(chat_id)}&text={message_str}"
    req = requests.post(req)
    if req.status_code != 200:
        logger.error(f"Error sending message to {chat_id}")
        print(f"Error sending message to {chat_id}")



#message = "Hello this is Meta-Man, a simple text sender for Meta trader Agent. I'll contact you once the agent needs more fund"

#if send_message(message):
#    send_message(message)
#    print("Message sent")
