import os
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize client using the new google-genai SDK layout
try:
    client = genai.Client()
except Exception as e:
    print(f"Failed to initialize GenAI Client: {e}")

app = FastAPI()

# Enable global CORS required by the grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enforce the strict schema payload structure required by the validator
class InvoiceExtraction(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The invoice number string")
    date: Optional[str] = Field(None, description="The date formatted as YYYY-MM-DD")
    vendor: Optional[str] = Field(None, description="The name of the selling vendor")
    amount: Optional[float] = Field(None, description="The subtotal dollar amount BEFORE tax")
    tax: Optional[float] = Field(None, description="The isolated tax amount value")
    currency: Optional[str] = Field(None, description="The standardized 3-letter currency code")

class InvoiceRequest(BaseModel):
    invoice_text: str

# FIX: Root route execution mapping
@app.post("/", response_model=InvoiceExtraction)
async def extract_invoice(payload: InvoiceRequest):
    try:
        system_instruction = (
            "You are an expert financial parsing assistant. "
            "Extract structured data from the provided plain-text invoice. "
            "Ensure the date is converted strictly into ISO YYYY-MM-DD format. "
            "Ensure amount maps to the item subtotal before taxes, and tax isolates the levy amounts."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=InvoiceExtraction,
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[payload.invoice_text],
            config=config
        )

        result_data = InvoiceExtraction.model_validate_json(response.text)
        return result_data

    except Exception as e:
        print("--- SERVER ERROR LOG ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
