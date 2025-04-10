import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# User registry
user_registry = {}  # chat_id -> username

def get_chat_id(username: str) -> int:
    for chat_id, user in user_registry.items():
        if user == username:
            return chat_id
    return None

# Telegram bot handlers
async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("hello function called")
    print(f"User: {update.effective_user.first_name}")
    await update.message.reply_text(f'Hello {update.effective_user.first_name} new code')
    print("Replied to user")

async def register(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id
    username = user.username or user.first_name

    # Register user
    user_registry[chat_id] = username
    print(f"Registered user: {username} with chat_id: {chat_id}")

    await update.message.reply_text(f"Registered {username} successfully!")

# Initialize and configure the telegram bot
def create_telegram_app():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("hello", hello))
    app.add_handler(CommandHandler("register", register))
    return app

# Start Telegram bot
async def run_telegram_bot(telegram_app):
    await telegram_app.initialize()
    await telegram_app.start()
    print("Telegram bot started")
    # Keeps the bot alive (like run_polling)
    await telegram_app.updater.start_polling()
    await telegram_app.updater.wait_until_closed() 