from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from helper import load_tokenizer, load_model, llm_output

load_dotenv()

app = FastAPI()

# Load model and tokenizer
model_name = "mehmetalpy/gpt2-fine-tuned"
tokenizer = load_tokenizer(model_name)
model = load_model(model_name)


# Define request and response models
class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    generated_text: str


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    input_text = request.text

    try:

        generated_text = llm_output(model, tokenizer, input_text)
        cleaned_text = ' '.join(generated_text.split())

    except Exception as e:
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during text generation.")

    return TextResponse(generated_text=cleaned_text)
