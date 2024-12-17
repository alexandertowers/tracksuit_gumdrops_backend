from openai import OpenAI
from config import settings

client = OpenAI(
    api_key=settings.openai_api_key
)


def generate_summary(terms, pos, neu, neg):
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""
                    You are an expert brand strategist. I have a set of customer reviews from a company. 
                    The top extracted terms from these reviews are: {terms}. 
                    The overall sentiment proportions are approximately: 
                    {pos*100:.0f}% positive, {neu*100:.0f}% neutral, and {neg*100:.0f}% negative.
                    """
            },
            {
                "role": "user",
                "content": """Please summarize what this company does and what its main value proposition is based on these terms. 
                    Then, comment on what aspects customers appreciate the most and where the company could improve, 
                    referencing the sentiment distribution.
                    Respond with a single concise paragraph.
                    Use a friendly tone and write as though you are speaking directly to the company's brand manager.
                    Also add 3 emojis at the end which you think describe the company's brand best.    
                """
            }
        ]
    )
    if completion.choices[0].message:
        return completion.choices[0].message
    else:
        return "Could not generate summary."