from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from fastapi import FastAPI, Request
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import io
import os
from dotenv import load_dotenv

load_dotenv() 

TOKEN = os.environ.get('TOKEN')
BOT_USERNAME = os.environ.get('BOT_USERNAME')
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# Configure Google API
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# FastAPI app
app = FastAPI()

# Global variable to hold the document text
text_chunks = []

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)  # Reset the file pointer to the beginning
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """You are a virtual Research Assistant. Your task is to answer questions related to research papers, 
    including details such as the title, abstract, keywords, field of research, and summary. Provide a 
    thorough and accurate response based on the provided context. Answer the question as detailed as 
    possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

async def process_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

# Telegram bot handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Please upload your research paper using the /upload command.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Here are the commands you can use:\n"
                                    "/start - Start interaction\n"
                                    "/upload - Upload your research paper (PDF)\n"
                                    "/abstract - Get the abstract of the paper\n"
                                    "/title - Get the title of the paper\n"
                                    "/summary - Get a summary of the paper\n"
                                    "/author - Get the authors of the paper\n"
                                    "/research_objective - Get the research objective of the paper\n"
                                    "/field_of_research - Get the field of research of the paper\n"
                                    "/query - Ask a question about the paper")

async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please send the PDF file of the research paper.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file = await document.get_file()
    file_path = f"/tmp/{document.file_name}"
    await file.download(file_path)

    with open(file_path, "rb") as f:
        pdf_docs = [f]
        raw_text = get_pdf_text(pdf_docs)
        global text_chunks
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    
    await update.message.reply_text("Document processed! You can now use the /abstract, /title, /summary commands, or ask a question using /query.")

async def abstract_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("What is the abstract of this paper?")
    await update.message.reply_text(response)

async def title_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("What is the title of this paper?")
    await update.message.reply_text(response)

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("Summarize the paper.")
    await update.message.reply_text(response)

async def author_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("Who are the authors of this paper?")
    await update.message.reply_text(response)

async def research_objective_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("What is the research objective of this paper?")
    await update.message.reply_text(response)

async def field_of_research_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await process_question("What is the field of research of this paper?")
    await update.message.reply_text(response)

async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = ' '.join(context.args)
    response = await process_question(user_question)
    await update.message.reply_text(response)

# Initialize the bot application
bot_app = Application.builder().token(TOKEN).build()

# Register command handlers
bot_app.add_handler(CommandHandler('start', start_command))
bot_app.add_handler(CommandHandler('help', help_command))
bot_app.add_handler(CommandHandler('upload', upload_command))
bot_app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
bot_app.add_handler(CommandHandler('abstract', abstract_command))
bot_app.add_handler(CommandHandler('title', title_command))
bot_app.add_handler(CommandHandler('summary', summary_command))
bot_app.add_handler(CommandHandler('author', author_command))
bot_app.add_handler(CommandHandler('research_objective', research_objective_command))
bot_app.add_handler(CommandHandler('field_of_research', field_of_research_command))
bot_app.add_handler(CommandHandler('query', query_command))

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
    # Initialize the bot application
    await bot_app.initialize()
    
    # Set the webhook for the bot
    await bot_app.bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def on_shutdown():
    await bot_app.shutdown()
