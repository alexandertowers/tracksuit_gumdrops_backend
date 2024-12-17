from fastapi import FastAPI, File, UploadFile,  HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from app.nlp_utils import process_file_for_tfidf
from app.openai_service import generate_summary
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from typing import List
import heapq
from pydantic import BaseModel

app = FastAPI()
download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class LLMRequest(BaseModel):
    terms: list[tuple[str, float]]
    sentiment: dict

# Allow CORS so the frontend (running on a different port) can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The port your frontend runs on
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode('utf-8')
    top_terms = process_file_for_tfidf(text)
    return {"terms": top_terms}

@app.post("/summarize")
async def summarize(terms: List[List]):
    # terms expected as a list of [term, score] pairs from frontend
    just_terms = [t[0] for t in terms]
    summary = generate_summary(just_terms)
    print(summary)
    return {"summary": summary}

@app.post("/llm")
def analyze_with_llm(payload: LLMRequest):
    summary = generate_summary(payload.terms, payload.sentiment["positive"], payload.sentiment["neutral"], payload.sentiment["negative"])
    return {"summary": summary}
    


@app.post("/sentiment")
async def analyze_sentiment(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode('utf-8')

        reviews = [line.strip() for line in text.split("\n") if line.strip()]
        if not reviews:
            raise HTTPException(status_code=400, detail="File is empty or improperly formatted.")

        sentiments = []
        for review in reviews:
            scores = sia.polarity_scores(review)
            sentiments.append({
                "review": review,
                "score": scores["compound"],
                "sentiment": (
                    "positive" if scores["compound"] > 0.05 else
                    "negative" if scores["compound"] < -0.05 else
                    "neutral"
                )
            })

        sentiment_counts = {
            "positive": sum(1 for s in sentiments if s["sentiment"] == "positive"),
            "negative": sum(1 for s in sentiments if s["sentiment"] == "negative"),
            "neutral": sum(1 for s in sentiments if s["sentiment"] == "neutral")
        }
        total_reviews = len(reviews)
        sentiment_proportions = {k: v / total_reviews for k, v in sentiment_counts.items()}

        most_positive = heapq.nlargest(3, sentiments, key=lambda x: x["score"])
        most_negative = heapq.nsmallest(3, sentiments, key=lambda x: x["score"])

        # Return results
        return {
            "sentiment_proportions": sentiment_proportions,
            "most_positive_reviews": most_positive,
            "most_negative_reviews": most_negative
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))