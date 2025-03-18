# chatbot.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from groq import Groq
from dotenv import load_dotenv
import json
import pysolr  # Import pysolr for Solr interaction

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SOLR_URL = "http://192.168.1.11:8983/solr/" # Solr URL from env, default if not set
SOLR_COLLECTION_NAME = "diamond_core" # Collection name from env, default if not set
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Embedding model name from env, default if not set

# Global embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------- Solr Client Initialization -------------------
def create_solr_client():
    return pysolr.Solr(f'{SOLR_URL}{SOLR_COLLECTION_NAME}', always_commit=False, timeout=10)

solr_client = create_solr_client()  # Initialize Solr client globally

# ------------------- Utility: Extract Constraints from Query -------------------
def extract_constraints_from_query(user_query):
    constraints = {}
    query_lower = user_query.lower()

    # ----- Style -----
    style_match = re.search(r'\b(lab\s*grown|lab|natural)\b', user_query, re.IGNORECASE)
    if style_match:
        style = style_match.group(1).lower()
        constraints["Style"] = "labgrown" if "lab" in style else "natural"

    # ----- Carat -----
    carat_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:carat[s]?|ct[s]?)\b', user_query, re.IGNORECASE)
    if carat_match:
        constraints["Carat"] = float(carat_match.group(1))

    # ----- Budget / Price Range -----
    price_range_match = re.search(r'between\s*\$?(\d+(?:,\d+)?)\s*(?:and|-)\s*\$?(\d+(?:,\d+)?)', user_query, re.IGNORECASE)
    if price_range_match:
        constraints["BudgetLow"] = float(price_range_match.group(1).replace(',', ''))
        constraints["BudgetHigh"] = float(price_range_match.group(2).replace(',', ''))
    else:
        #"around" or "close to"
        approx_price_match = re.search(r'\b(?:around|close to)\s*\$?(\d+(?:,\d+)?)', user_query, re.IGNORECASE)
        if approx_price_match:
            constraints["BudgetTarget"] = float(approx_price_match.group(1).replace(',', ''))
        else:
            #support phrases like "under 5000", "at price 5000", "price 5000"
            pattern1 = r'\b(?:under|at (?:price|cost|value)|(?:price|cost|value))\s*\$?(\d+(?:,\d+)?)(?:\$)?\b'
            pattern2 = r'\$?(\d+(?:,\d+)?)(?:\$)?\s*(?:price|cost|value)\b'
            budget_match = re.search(pattern1, user_query, re.IGNORECASE) or re.search(pattern2, user_query, re.IGNORECASE)
            if budget_match:
                budget_str = budget_match.group(1).replace(',', '')
                constraints["Budget"] = float(budget_str)


    # ----- Color -----
    color_mapping = {
        "f light blue": "f",
        "g light": "g",
        "j faint green": "j",
        "j very light blue": "j",
        "k faint brown": "k",
        "k faint color": "k",
        "m faint brown": "m",
        "n v light brown": "n",
        "l faint brown": "l",
        "n very light yellow": "n",
        "n very light brown": "n",
        "g light green": "g"
    }
    found_color = False
    for desc, letter in color_mapping.items():
        if re.search(r'\b' + re.escape(desc) + r'\b', query_lower):
            constraints["Color"] = letter
            found_color = True
            break
    if not found_color:
        simple_color_match = re.search(r'\b([defghijklmn])\b', query_lower, re.IGNORECASE)
        if simple_color_match:
            constraints["Color"] = simple_color_match.group(1).lower()

    # ----- Clarity -----
    clarity_match = re.search(r'\b(if|vvs1|vvs2|vs1|vs2|si1|si2)\b', user_query, re.IGNORECASE)
    if clarity_match:
        constraints["Clarity"] = clarity_match.group(1).lower()

    # ----- Quality Attributes for Cut, Polish, and Symmetry -----
    quality_mapping = {
        "ex": "ex",
        "excellent": "ex",
        "id": "id",
        "ideal": "id",
        "vg": "vg",
        "very good": "vg",
        "good": "vg",
        "gd": "gd",
        "f": "f",
        "p": "p",
        "fr": "fr"
    }
    quality_pattern_cut_polish = r'(?:{attr}\s*(?:is\s*)?((?:ex|excellent|id|ideal|vg|very good|good|gd|f|p)))|(?:(?:(ex|excellent|id|ideal|vg|very good|good|gd|f|p))\s*{attr})'
    quality_pattern_symmetry = r'(?:symmetry\s*(?:is\s*)?((?:ex|excellent|id|ideal|vg|very good|good|gd|f|p|fr)))|(?:(?:(ex|excellent|id|ideal|vg|very good|good|gd|f|p|fr))\s*symmetry)'

    # ----- Cut -----
    cut_regex = quality_pattern_cut_polish.format(attr='cut')
    cut_match = re.search(cut_regex, user_query, re.IGNORECASE)
    if cut_match:
        quality = (cut_match.group(1) or cut_match.group(2)).lower()
        constraints["Cut"] = quality_mapping.get(quality, quality)

    # ----- Polish -----
    polish_regex = quality_pattern_cut_polish.format(attr='polish')
    polish_match = re.search(polish_regex, user_query, re.IGNORECASE)
    if polish_match:
        quality = (polish_match.group(1) or polish_match.group(2)).lower()
        constraints["Polish"] = quality_mapping.get(quality, quality)

    # ----- Symmetry -----
    symmetry_match = re.search(quality_pattern_symmetry, user_query, re.IGNORECASE)
    if symmetry_match:
        quality = (symmetry_match.group(1) or symmetry_match.group(2)).lower()
        constraints["Symmetry"] = quality_mapping.get(quality, quality)

    # ----- Fluorescence (Flo) -----
    if re.search(r'\b(flo|fluorescence)\b', user_query, re.IGNORECASE):
        if re.search(r'\b(no|none|non)\b', query_lower):
            constraints["Flo"] = "non"
        elif re.search(r'\bfaint\b', query_lower):
            constraints["Flo"] = "fnt"
        elif re.search(r'\bmedium\b', query_lower):
            constraints["Flo"] = "med"
        elif re.search(r'\bvery slight\b', query_lower):
            constraints["Flo"] = "vsl"
        elif re.search(r'\bvery strong\b', query_lower):
            constraints["Flo"] = "vst"
        elif re.search(r'\bslight\b', query_lower):
            constraints["Flo"] = "slt"
        else:
            constraints["Flo"] = "fnt"

    # ----- Lab -----
    lab_options = ['igi', 'gia', 'gcal', 'none', 'gsi', 'hrd', 'sgl', 'other', 'egl', 'ags', 'dbiod']
    for lab in lab_options:
        if re.search(r'\b' + re.escape(lab) + r'\b', query_lower):
            constraints["Lab"] = lab
            break

    # ----- Shape -----
    shape_options = [
        'cushion modified', 'round-cornered rectangular modified brilliant', 'old european brilliant',
        'butterfly modified brilliant', 'old mine brilliant', 'modified rectangular brilliant', 'cushion brilliant',
        'square emerald', 'european cut', 'square radiant', 'old miner', 'cushion', 'triangular', 'square',
        'old european', 'asscher', 'princess', 'oval', 'round', 'pear', 'emerald', 'marquise', 'radiant',
        'heart', 'baguette', 'octagonal', 'shield', 'hexagonal', 'other', 'half moon', 'rose',
        'trapeze', 'trapezoid', 'trilliant', 'lozenge', 'kite', 'pentagonal', 'tapered baguette',
        'pentagon', 'heptagonal', 'rectangular', 'bullet', 'briollette', 'rhomboid', 'others', 'star',
        'calf', 'nonagonal'
    ]
    shape_options = sorted([s.lower() for s in shape_options], key=len, reverse=True)
    for shape in shape_options:
        if re.search(r'\b' + re.escape(shape) + r'\b', query_lower):
            constraints["Shape"] = shape
            break

    # ----- Price Ordering Preference (if no explicit budget) -----
    if "Budget" not in constraints:
        if any(keyword in query_lower for keyword in ["cheapest", "lowest price", "affordable", "low budget"]):
            constraints["PriceOrder"] = "asc"
        elif any(keyword in query_lower for keyword in ["most expensive", "highest price", "priciest", "expensive", "high budget"]):
            constraints["PriceOrder"] = "desc"

    return constraints

# ------------------- Re-Ranking Function -------------------
def rerank_results(results_df, constraints):
    """
    Compute a combined score for each diamond based on weighted differences from the desired Carat and Price.
    Lower scores indicate a closer match.
    """
    # Define weights for attributes
    carat_weight = 0.5 if "Carat" in constraints else 0.0
    price_weight = 0.5 if ("Budget" in constraints or 
                             ("BudgetLow" in constraints and "BudgetHigh" in constraints) or 
                             "BudgetTarget" in constraints) else 0.0

    if carat_weight + price_weight == 0:
        return results_df  # Nothing to re-rank

    def compute_score(row):
        score = 0.0
        # Carat difference (normalized)
        if "Carat" in constraints and row.get("Carat") is not None:
            desired_carat = constraints["Carat"]
            carat_diff = abs(row["Carat"] - desired_carat) / desired_carat
            score += carat_weight * carat_diff

        # Price difference:
        if "BudgetTarget" in constraints and row.get("Price") is not None:
            # For approximate queries like "price around 8000$",
            # use the absolute normalized difference from the target.
            target = constraints["BudgetTarget"]
            price_diff = abs(row["Price"] - target) / target
            score += price_weight * price_diff
        elif "BudgetLow" in constraints and "BudgetHigh" in constraints and row.get("Price") is not None:
            # If a range is provided, use the midpoint as target
            target = (constraints["BudgetLow"] + constraints["BudgetHigh"]) / 2
            price_diff = abs(row["Price"] - target) / target
            score += price_weight * price_diff
        elif "Budget" in constraints and row.get("Price") is not None:
            desired_budget = constraints["Budget"]
            # Set a lower bound (e.g., 50% of the budget) for what is considered a "good" price.
            lower_bound = 0.5 * desired_budget
            # If the diamond's price is below the lower bound, assign maximum penalty.
            if row["Price"] < lower_bound:
                price_diff = 1.0
            else:
                # Compute a normalized difference where 0 means price equals the desired_budget
                # and 1 means price equals the lower_bound.
                price_diff = (desired_budget - row["Price"]) / (desired_budget - lower_bound)
            score += price_weight * price_diff
        return score

    results_df["combined_score"] = results_df.apply(compute_score, axis=1)
    return results_df.sort_values("combined_score", ascending=True)



# ------------------- Hybrid Search with Solr -------------------
def hybrid_search(user_query, solr_client, top_k=10):
    """
    Hybrid search that uses extracted constraints to build filter queries.
    This version adds filters for Style, Carat, Clarity, Color, and Cut.
    """
    constraints = extract_constraints_from_query(user_query)
    solr_query = "*:*"  # Base query: match all documents
    filter_queries = []

    # Style filter
    if "Style" in constraints:
        style_value = constraints['Style'].lower()
        if style_value == "labgrown":
            style_value = "lab"  # Convert labgrown to lab if needed
        filter_queries.append(f"Style:({style_value})")

    # Carat filter: ±10% tolerance around the queried value
    if "Carat" in constraints:
        carat_val = constraints["Carat"]
        tolerance = 0.1 * carat_val
        filter_queries.append(f"Carat:[{carat_val - tolerance} TO {carat_val + tolerance}]")

    # Clarity filter
    if "Clarity" in constraints:
        clarity_val = constraints["Clarity"]
        filter_queries.append(f"Clarity:({clarity_val.upper()})")

    # Color filter
    if "Color" in constraints:
        color_val = constraints["Color"]
        filter_queries.append(f"Color:({color_val.upper()})")

    # Cut filter
    if "Cut" in constraints:
        cut_val = constraints["Cut"]
        filter_queries.append(f"Cut:({cut_val.upper()})")

    # 
    # Budget filter: use range if provided, otherwise use single budget value
    if "BudgetLow" in constraints and "BudgetHigh" in constraints:
        filter_queries.append(f"Price:[{constraints['BudgetLow']} TO {constraints['BudgetHigh']}]")
    elif "BudgetTarget" in constraints:
        target = constraints["BudgetTarget"]
        tolerance = 0.2 * target  # You might try reducing this to 0.15 if needed.
        filter_queries.append(f"Price:[{target - tolerance} TO {target + tolerance}]")


    query_embedding = model.encode(user_query, convert_to_numpy=True).tolist()
    knn_query = f"{{!knn f=vector topK={top_k}}}{query_embedding}"

    query_params = {
        "q": knn_query,
        "fq": filter_queries,
        "fl": "Carat,Clarity,Color,Cut,Shape,Price,Style,Polish,Symmetry,Lab,Flo,pdf,image,video,score,dist",
        "rows": top_k
    }

    try:
        results = solr_client.search(**query_params)
        # (Optional debugging: print raw results)
        if not results.docs:
            print("No documents found in Solr results.")
        results_df = pd.DataFrame([dict(doc) for doc in results.docs])
        if not results_df.empty:
            if 'score' in results_df.columns:
                results_df['distance'] = 1.0 - results_df['score']
            elif 'dist' in results_df.columns:
                results_df['distance'] = results_df['dist']
        return results_df

    except pysolr.SolrError as e:
        print(f"Solr search error: {e}")
        return pd.DataFrame()

# ------------------- Groq Integration (No Changes) -------------------
def generate_groq_response(user_query, relevant_data, client):
    prompt = f"""
You are a friendly, expert diamond consultant with years of experience helping customers find the perfect diamond.
Your response should be personal, warm, and engaging. Provide an expert recommendation based on the customer's query.

Please analyze the following diamond details and produce a JSON response that includes the top matching diamonds.
Your response should include:
1. A brief introductory paragraph (one or two sentences) written in a conversational tone explaining what you found. Include an expert recommendation for a top pick, explaining why that diamond stands out.
2. A special marker <diamond-data> immediately followed by a valid JSON array of diamond objects.
3. Close with </diamond-data>.

Each diamond object must include the following attributes:
- Carat
- Clarity
- Color
- Cut
- Shape
- Price
- Style
- Polish
- Symmetry
- Lab
- Flo
- pdf
- image
- video

For example, your response should look like:
"Hi there! Based on your query for a 4 carat diamond with VS1 clarity, I recommend the second diamond for its excellent balance between carat and clarity—it truly stands out. (Also compare the diamond data and say why which option is better) Here are the best options:
<diamond-data>
[
    {{
        "Carat": 4.0,
        "Clarity": "VS1",
        "Color": "F",
        "Cut": "EX",
        "Shape": "ROUND",
        "Price": "5000",
        "Style": "labgrown",
        "Polish": "EX",
        "Symmetry": "EX",
        "Lab": "IGI",
        "Flo": "NON"
    }},
    ...more diamonds...
]
</diamond-data>"

Below are some diamond details that might be relevant:
{relevant_data}

Make sure the JSON is valid and can be parsed by JavaScript's JSON.parse() function.
"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=750
    )
    return chat_completion.choices[0].message.content

def convert_markdown_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# ------------------- Main Chatbot Logic -------------------
def diamond_chatbot(user_query, solr_client, client): # Modified to take solr_client, remove df, faiss, model
    """
    Handles the chatbot's logic using Solr for search.
    """
    # Handle greetings (no change)
    if user_query.strip().lower() in ["hi", "hello"]:
        return "Hey there! I'm your diamond guru 😎. Ready to help you find that perfect sparkle? Tell me what you're looking for!"

    # Extract constraints from the user query (no change)
    constraints = extract_constraints_from_query(user_query)

    # Only fall back if there are no constraints AND no ordering keywords in the query. (no change)
    if not constraints and not any(keyword in user_query.lower() for keyword in ["maximum", "minimum", "lowest", "highest", "largest", "smallest"]):
        return "Hello! I'm your diamond assistant. Please let me know your preferred carat, clarity, color, cut, or budget so I can help you find the perfect diamond."

    # Proceed with searching for diamonds using Solr
    results_df = hybrid_search(user_query, solr_client, top_k=200) # Pass solr_client, remove df, faiss, model
    if results_df.empty:
        return "No matching diamonds found. Please try a different query."

    # Select top 5 matching diamonds (no change - still uses DataFrame)
    top_5 = results_df.head(5)
    relevant_data_list = []
    for index, row in top_5.iterrows():
        diamond_info = {
            "Carat": row.get("Carat"),
            "Clarity": row.get("Clarity"),
            "Color": row.get("Color"),
            "Cut": row.get("Cut"),
            "Shape": row.get("Shape"),
            "Price": row.get("Price"),
            "Style": row.get("Style"),
            "Polish": row.get("Polish"),
            "Symmetry": row.get("Symmetry"),
            "Lab": row.get("Lab"),
            "Flo": row.get("Flo"),
            "Certificate": row.get("pdf"),
            "Image": row.get("Image"),
            "Video": row.get("Video")
        }
        relevant_data_list.append(diamond_info)
    relevant_data_json = json.dumps(relevant_data_list, indent=2) # Convert to JSON string
    relevant_data = relevant_data_json # Use JSON string as relevant_data

    # Generate response using Groq AI (no change)
    groq_response = generate_groq_response(user_query, relevant_data, client)

    return groq_response

# ------------------- Main Execution -------------------
def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq()

    solr_client = create_solr_client() # Initialize Solr client here

    # Conversation loop (minor changes - removed data loading)
    while True:
        user_query = input("Hi! How can I help you? : ")
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for visiting! Have a wonderful day.")
            break

        # Process greetings (no change)
        if user_query.strip().lower() in ["hi", "hello"]:
            response = diamond_chatbot(user_query, solr_client, client) # Pass solr_client
            print(response)
            print("\n---\n")
            continue

        constraints = extract_constraints_from_query(user_query)
        if "Style" not in constraints:
            style_input = input("Please specify the style (LabGrown or Natural): ")
            user_query += " " + style_input

        response = diamond_chatbot(user_query, solr_client, client) # Pass solr_client
        print(response)
        print("\n---\n")

if __name__ == "__main__":
    main()