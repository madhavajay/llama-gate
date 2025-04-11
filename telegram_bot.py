import os
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import re
from typing import Optional
from models import RequestState  # Import RequestState from models

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

# Dictionary to store pending approvals
pending_approvals = {}  # chat_id -> (request_id, query, action)

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

def find_request_by_id_or_query(search_term: str) -> Optional[tuple]:
    """Find a request by ID or query content."""
    from queue_storage import QueueStorage
    request_storage = QueueStorage("storage/requests")
    
    # Search in storage
    for item in request_storage.list_items():
        if search_term.lower() in item.id.lower() or search_term.lower() in item.query.lower():
            return (item.id, item.query)
    
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

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text("Please provide a request ID or query to approve. Usage: /approve <request_id_or_query>")
        return
    
    search_term = " ".join(context.args)
    request_info = find_request_by_id_or_query(search_term)
    
    if not request_info:
        await update.message.reply_text("No matching request found.")
        return
    
    request_id, query = request_info
    pending_approvals[chat_id] = (request_id, query, "approve")  # Add action type
    
    await update.message.reply_text(
        f"Found request:\nID: {request_id}\nQuery: {query}\n\n"
        "Would you like to approve this request? (y/n)"
    )

async def reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text("Please provide a request ID or query to reject. Usage: /reject <request_id_or_query>")
        return
    
    search_term = " ".join(context.args)
    request_info = find_request_by_id_or_query(search_term)
    
    if not request_info:
        await update.message.reply_text("No matching request found.")
        return
    
    request_id, query = request_info
    pending_approvals[chat_id] = (request_id, query, "reject")  # Add action type
    
    await update.message.reply_text(
        f"Found request:\nID: {request_id}\nQuery: {query}\n\n"
        "Would you like to reject this request? (y/n)"
    )

async def handle_approval_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    
    if chat_id not in pending_approvals:
        return
    
    response = update.message.text.lower()
    request_id, query, action = pending_approvals[chat_id]
    
    if response == 'y':
        # Process the request based on action type
        from queue_storage import QueueStorage
        request_storage = QueueStorage("storage/requests")
        
        if action == "approve":
            request_storage.update_item_state(request_id, RequestState.APPROVED)
            await update.message.reply_text(f"Request approved:\nID: {request_id}\nQuery: {query}")
        else:  # reject
            request_storage.update_item_state(request_id, RequestState.REJECTED)
            await update.message.reply_text(f"Request rejected:\nID: {request_id}\nQuery: {query}")
    elif response == 'n':
        await update.message.reply_text("Action cancelled.")
    else:
        await update.message.reply_text("Please respond with 'y' for yes or 'n' for no.")
        return
    
    # Clear the pending approval
    del pending_approvals[chat_id]

async def list_requests(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all pending requests."""
    from queue_storage import QueueStorage
    request_storage = QueueStorage("storage/requests")
    
    # Get all items and filter for pending ones
    pending_requests = [
        item for item in request_storage.list_items()
        if item.state == RequestState.PENDING
    ]
    
    if not pending_requests:
        await update.message.reply_text("No pending requests found.")
        return
    
    # Format the message
    message = "Pending requests:\n\n"
    for i, request in enumerate(pending_requests, 1):
        message += f"{i}. ID: {request.id}\n   Query: {request.query}\n\n"
    
    await update.message.reply_text(message)

# Initialize and configure the telegram bot
def create_telegram_app():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("hello", hello))
    app.add_handler(CommandHandler("register", register))
    app.add_handler(CommandHandler("approve", approve))
    app.add_handler(CommandHandler("reject", reject))
    app.add_handler(CommandHandler("list", list_requests))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_approval_response))
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