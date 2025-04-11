import os
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# User registry file path
USER_REGISTRY_FILE = "user_registry.json"

# Load user registry from file
def load_user_registry():
    if os.path.exists(USER_REGISTRY_FILE):
        with open(USER_REGISTRY_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user registry to file
def save_user_registry():
    with open(USER_REGISTRY_FILE, "w") as file:
        json.dump(user_registry, file)

# User registry
user_registry = load_user_registry()  # chat_id -> username

def get_first_registered_chat_id() -> int:
    """Get the chat ID of the first registered user."""
    if not user_registry:
        return None
    return next(iter(user_registry.keys()))

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
    save_user_registry()
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

# Function to send notification
async def send_notification(telegram_app, message: str) -> bool:
    """Send a notification to the first registered user."""
    chat_id = get_first_registered_chat_id()
    if not chat_id:
        print("No registered users to send notification to")
        return False
    
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=message)
        return True
    except Exception as e:
        print(f"Error sending notification: {e}")
        return False 