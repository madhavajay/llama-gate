import os
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import re
from typing import Optional
from models import RequestState  # Import RequestState from models
from shared_state import pending_approvals, user_registry  # Import shared state

# Load environment variables
load_dotenv()

# User registry file path
USER_REGISTRY_FILE = "user_registry.json"

# Load user registry from file
def load_user_registry():
    if os.path.exists(USER_REGISTRY_FILE):
        with open(USER_REGISTRY_FILE, "r") as file:
            # Convert string keys to integers when loading from JSON
            return {int(k): v for k, v in json.load(file).items()}
    return {}

# Save user registry to file
def save_user_registry():
    # Convert integer keys to strings when saving to JSON
    with open(USER_REGISTRY_FILE, "w") as file:
        json.dump({str(k): v for k, v in user_registry.items()}, file)

# Initialize user registry
user_registry.update(load_user_registry())

def get_first_registered_chat_id() -> int:
    """Get the chat ID of the first registered user."""
    if not user_registry:
        return None
    return int(next(iter(user_registry.keys())))  # Ensure chat_id is an integer

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
    chat_id = int(update.effective_chat.id)  # Ensure chat_id is an integer
    username = user.username or user.first_name

    # Register user
    user_registry[chat_id] = username
    save_user_registry()
    print(f"Registered user: {username} with chat_id: {chat_id}")

    await update.message.reply_text(f"Registered {username} successfully!")

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id)  # Ensure chat_id is an integer
    
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
        "Would you like to:\n"
        "y - Approve and run\n"
        "r - Run without approving\n"
        "n - Cancel"
    )

async def reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id)  # Ensure chat_id is an integer
    
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
    chat_id = int(update.effective_chat.id)  # Ensure chat_id is an integer
    user_response = update.message.text.lower()
    
    print(f"\n[TELEGRAM_BOT] Received response: {user_response} from chat_id: {chat_id}")
    print(f"[TELEGRAM_BOT] Current pending_approvals: {pending_approvals}")
    
    if chat_id not in pending_approvals:
        print(f"[TELEGRAM_BOT] No pending approval found for chat_id: {chat_id}")
        await update.message.reply_text("No pending approval found.")
        return
    
    request_id, query, action = pending_approvals[chat_id]
    print(f"[TELEGRAM_BOT] Found pending approval: request_id={request_id}, action={action}")
    
    # Update the request in the queue
    from queue_storage import QueueStorage
    request_storage = QueueStorage("storage/requests")
    
    # Get the current request
    request = request_storage.load_item(request_id)
    if not request:
        print(f"[TELEGRAM_BOT] Could not find request {request_id} in storage")
        await update.message.reply_text("Request not found in storage.")
        del pending_approvals[chat_id]
        return
    
    print(f"[TELEGRAM_BOT] Found request in storage, current state: {request.state}")
    
    # For approve_with_result, only accept y/n
    if action == "approve_with_result":
        print(f"[TELEGRAM_BOT] Handling approve_with_result action")
        if user_response not in ['y', 'n']:
            print(f"[TELEGRAM_BOT] Invalid response for approve_with_result: {user_response}")
            await update.message.reply_text("Please respond with 'y' or 'n'.")
            return
        
        if user_response == 'y':
            print(f"[TELEGRAM_BOT] Setting request {request_id} to APPROVED")
            request.state = RequestState.APPROVED
            await update.message.reply_text("Request approved.")
        else:  # n
            print(f"[TELEGRAM_BOT] Setting request {request_id} to REJECTED")
            request.state = RequestState.REJECTED
            await update.message.reply_text("Request rejected.")
        
        print(f"[TELEGRAM_BOT] Saving request with new state: {request.state}")
        request_storage.save_item(request)
        print(f"[TELEGRAM_BOT] Removing pending approval for chat_id: {chat_id}")
        del pending_approvals[chat_id]
        print(f"[TELEGRAM_BOT] Updated pending_approvals: {pending_approvals}")
        return
    
    # For regular approvals, accept y/r/n
    if user_response not in ['y', 'n', 'r']:
        print(f"[TELEGRAM_BOT] Invalid response for regular approval: {user_response}")
        await update.message.reply_text("Please respond with 'y', 'r', or 'n'.")
        return
    
    if user_response == 'n':
        print(f"[TELEGRAM_BOT] Cancelling action for request {request_id}")
        del pending_approvals[chat_id]
        await update.message.reply_text("Action cancelled.")
        return
    
    # If request is already approved, just set can_run and proceed
    if request.state == RequestState.APPROVED:
        print(f"[TELEGRAM_BOT] Request {request_id} is already approved, setting can_run=True")
        request.can_run = True
        request_storage.save_item(request)
        del pending_approvals[chat_id]
        await update.message.reply_text("Request is already approved and will be processed.")
        return
    
    request.can_run = True
    if user_response == 'y':
        print(f"[TELEGRAM_BOT] Setting request {request_id} to APPROVED")
        request.state = RequestState.APPROVED
        await update.message.reply_text("Request approved and will be processed.")
    else:  # r
        print(f"[TELEGRAM_BOT] Setting request {request_id} to can_run=True without approval")
        await update.message.reply_text("Request will be processed without approval.")
    
    print(f"[TELEGRAM_BOT] Saving request with new state: {request.state}, can_run: {request.can_run}")
    request_storage.save_item(request)
    
    print(f"[TELEGRAM_BOT] Removing pending approval for chat_id: {chat_id}")
    del pending_approvals[chat_id]
    print(f"[TELEGRAM_BOT] Updated pending_approvals: {pending_approvals}")

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

async def force_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve all pending requests and allow them to execute."""
    from queue_storage import QueueStorage
    request_storage = QueueStorage("storage/requests")
    
    print(f"\n[FORCE_APPROVE] Starting force approve process")
    
    # Get all items and filter for pending ones
    all_items = request_storage.list_items()
    print(f"[FORCE_APPROVE] Found {len(all_items)} total items in storage")
    
    pending_requests = [
        item for item in all_items
        if item.state == RequestState.PENDING
    ]
    
    print(f"[FORCE_APPROVE] Found {len(pending_requests)} pending requests")
    
    if not pending_requests:
        await update.message.reply_text("No pending requests found.")
        return
    
    # Approve all pending requests
    approved_count = 0
    for request in pending_requests:
        print(f"[FORCE_APPROVE] Processing request: {request.id}")
        print(f"[FORCE_APPROVE] Current state: {request.state}")
        
        request.state = RequestState.APPROVED
        request.can_run = True
        request_storage.save_item(request)
        approved_count += 1
        
        print(f"[FORCE_APPROVE] Updated state: {request.state}, can_run: {request.can_run}")
    
    print(f"[FORCE_APPROVE] Completed. Approved {approved_count} requests")
    await update.message.reply_text(f"Approved {approved_count} pending requests. They will now be processed.")

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
    app.add_handler(CommandHandler("force_approve", force_approve))
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