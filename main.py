from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from fastapi import FastAPI, Request
import uvicorn

TOKEN = '6522756388:AAHZFrOgeCyf8T7q3YZ-zJPenVvPAUmzvP0'
BOT_USERNAME = '@researchA_bot'
WEBHOOK_URL = 'https://rabot50.onrender.com/webhook'

app = FastAPI()

# Define your bot commands and handlers as before
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Thanks for chatting with me!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I am a Python bot. Please type something so I can respond!")

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is a custom command!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Your message handling logic goes here
    await update.message.reply_text(f"Echo: {update.message.text}")

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

# Initialize the bot application
bot_app = Application.builder().token(TOKEN).build()

# Register command handlers
bot_app.add_handler(CommandHandler('start', start_command))
bot_app.add_handler(CommandHandler('help', help_command))
bot_app.add_handler(CommandHandler('custom', custom_command))

# Register message handler
bot_app.add_handler(MessageHandler(filters.TEXT, handle_message))

# Register error handler
bot_app.add_error_handler(error)

@app.post('/webhook')
async def webhook_handler(request: Request):
    update = await request.json()
    update = Update.de_json(update, bot_app.bot)
    await bot_app.process_update(update)
    return 'OK'

@app.on_event("startup")
async def on_startup():
    await bot_app.bot.set_webhook(WEBHOOK_URL)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
