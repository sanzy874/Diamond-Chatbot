import pandas as pd
import numpy as np
import faiss
import re
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Global: Paths for prepared embeddings, FAISS index, source CSV, and model folder
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'
FAISS_INDEX_FILE_PATH = 'diamond_faiss_index.faiss'
DIAMONDS_CSV_FILE = '/home/sandra/Internship/Chat Bot/Gemma/diamonds.csv'
MODEL_PATH = 'sentence_transformer_model'

# ------------------- Load Data & FAISS Index -------------------
def load_data_and_index(embedding_file, faiss_index_file, csv_file, model_path):
    # Load the original diamonds data from CSV
    df = pd.read_csv(csv_file)
    df = df.replace({r'[^\x00-\x7F]+': ''}, regex=True)
    # Convert all values to lowercase
    df = df.apply(lambda x: x.astype(str).str.lower())
    print(f"Number of rows in dataset: {df.shape[0]}")
    print(f"Column names in dataset: {df.columns.tolist()}")

    # Create a combined text field for semantic search
    df['combined_text'] = (
        "Style: " + df['style'].astype(str) + ", " +
        "Carat: " + df['carat'].astype(str) + ", " +
        "Clarity: " + df['clarity'].astype(str) + ", " +
        "Color: " + df['color'].astype(str) + ", " +
        "Cut: " + df['cut'].astype(str) + ", " +
        "Shape: " + df['shape'].astype(str) + ", " +
        "Price: " + df['price'].astype(str) + ", " +
        "Lab: " + df['lab'].astype(str) + ", " +
        "Polish: " + df['polish'].astype(str) + ", " +
        "Symmetry: " + df['symmetry'].astype(str) + ", " +
        "Fluorescence: " + df['flo'].astype(str) + ", " +
        "Length: " + df['length'].astype(str) + ", " +
        "Width: " + df['width'].astype(str) + ", " +
        "Depth: " + df['depth'].astype(str) + ", " +
        "Certificate: " + df['pdf'].astype(str)
    )

    # Ensure Carat is numeric
    df["carat"] = pd.to_numeric(df["carat"], errors="coerce")

    # Load precomputed embeddings
    embeddings = np.load(embedding_file)

    # Load or rebuild FAISS index
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
    else:
        print("FAISS index not found. Rebuilding from embeddings...")
        embedding_dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dimension)
        index.add(embeddings)
        faiss.write_index(index, faiss_index_file)

    # Load the SentenceTransformer model (needed to encode the user query)
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer('all-mpnet-base-v2')

    print("Loaded data, embeddings, FAISS index, and model from disk.")
    return df, embeddings, index, model

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
    carat_match = re.search(r'(\d+(\.\d+)?)\s*-?\s*carat', user_query, re.IGNORECASE)
    if carat_match:
        constraints["Carat"] = float(carat_match.group(1))

    # ----- Budget -----
    pattern1 = r'\b(?:under|at price|price)\s*\$?(\d+(?:,\d+)?)(?:\$)?\b'
    pattern2 = r'\$?(\d+(?:,\d+)?)(?:\$)?\s*price\b'
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

    # ----- Price Ordering Preference -----
    if "Budget" not in constraints:
        if any(keyword in query_lower for keyword in ["cheapest", "lowest price", "affordable", "low budget"]):
            constraints["PriceOrder"] = "asc"
        elif any(keyword in query_lower for keyword in ["most expensive", "highest price", "priciest", "expensive", "high budget"]):
            constraints["PriceOrder"] = "desc"

    return constraints

# ------------------- Hybrid Search (Semantic + Filter + Composite Ranking) -------------------
def hybrid_search(user_query, df, faiss_index, model, top_k=200):
    constraints = extract_constraints_from_query(user_query)

    # ----- Filter by Style -----
    if "Style" in constraints:
        style_val = constraints["Style"].lower()
        df = df[df['style'].str.lower().str.contains(style_val)]
        if df.empty:
            print("No diamonds found for the specified style.")
            return pd.DataFrame()

    # ----- Filter by Shape -----
    if "Shape" in constraints:
        df = df[df["shape"].str.lower().str.contains(constraints["Shape"].lower())]
        if df.empty:
            print(f"No {constraints['Shape']} diamonds found.")
            return pd.DataFrame()

    # ----- Filter by Clarity -----
    if "Clarity" in constraints:
        clarity_regex = rf'^{re.escape(constraints["Clarity"].lower())}$'
        df = df[df["clarity"].str.lower().str.match(clarity_regex)]
        if df.empty:
            print(f"No diamonds found with clarity {constraints['Clarity']}.")
            return pd.DataFrame()

    # ----- Filter by Color -----
    if "Color" in constraints:
        color_regex = rf'^{re.escape(constraints["Color"].lower())}\b'
        df = df[df["color"].str.lower().str.match(color_regex)]
        if df.empty:
            print(f"No diamonds found with color {constraints['Color']}.")
            return pd.DataFrame()

    # ----- Filter by Lab -----
    if "Lab" in constraints:
        df = df[df["lab"].str.lower() == constraints["Lab"].lower()]
        if df.empty:
            print(f"No diamonds found with lab certificate {constraints['Lab']}.")
            return pd.DataFrame()

    # ----- Filter by Fluorescence (Flo) -----
    if "Flo" in constraints:
        df = df[df["flo"].str.lower() == constraints["Flo"].lower()]
        if df.empty:
            print(f"No diamonds found with fluorescence {constraints['Flo']}.")
            return pd.DataFrame()

    # ----- Filter by Budget -----
    if "Budget" in constraints:
        user_budget = constraints["Budget"]
        df = df[df["price"].astype(float) <= user_budget]
        if df.empty:
            print(f"No diamonds found under price {user_budget}.")
            return pd.DataFrame()

    # ----- Strict Filtering for Quality Attributes (Cut, Polish, Symmetry) -----
    quality_attrs = ["Cut", "Polish", "Symmetry"]
    specified_quality = [attr for attr in quality_attrs if attr in constraints]
    if len(specified_quality) >= 2:
        for attr in specified_quality:
            df = df[df[attr.lower()].str.lower() == constraints[attr].lower()]
        if df.empty:
            print(f"No diamonds found that exactly match the specified {', '.join(specified_quality)} criteria.")
            return pd.DataFrame()

    # ----- Handling Carat: If not specified, fallback sorting -----
    if "Carat" not in constraints:
        if "PriceOrder" in constraints and constraints["PriceOrder"] == "asc":
            results_df = df.sort_values(by="price", ascending=True)
        elif any(word in user_query.lower() for word in ["minimum", "lowest", "smallest"]):
            results_df = df.sort_values(by="carat", ascending=True)
        else:
            results_df = df.sort_values(by="price", ascending=False)
        return results_df.head(5).reset_index(drop=True)

    # ----- Carat-specific Filtering using FAISS -----
    tolerance = 0.01 if constraints.get("Style", "").lower() == "labgrown" else 0.05
    df_carat = df[(df['carat'] >= constraints["Carat"] - tolerance) & (df['carat'] <= constraints["Carat"] + tolerance)]
    if df_carat.empty:
        relaxed_tolerance = tolerance * 2
        df_carat = df[(df['carat'] >= constraints["Carat"] - relaxed_tolerance) & (df['carat'] <= constraints["Carat"] + relaxed_tolerance)]
    if not df_carat.empty:
        subset_indices = df_carat.index.tolist()
        all_embeddings = np.load(EMBEDDING_FILE_PATH)
        subset_embeddings = all_embeddings[subset_indices]
        temp_index = faiss.IndexFlatL2(all_embeddings.shape[1])
        temp_index.add(subset_embeddings)
        new_top_k = min(top_k, len(df_carat))
        query_embedding = model.encode(user_query, convert_to_numpy=True)
        D, I = temp_index.search(np.array([query_embedding]), new_top_k)
        valid_indices = [i for i in I[0] if 0 <= i < len(df_carat)]
        valid_D = D[0][:len(valid_indices)]
        results_df = df_carat.iloc[valid_indices].copy()
        results_df['distance'] = valid_D
    else:
        query_embedding = model.encode(user_query, convert_to_numpy=True)
        new_top_k = min(top_k, df.shape[0])
        D, I = faiss_index.search(np.array([query_embedding]), new_top_k)
        valid_indices = [i for i in I[0] if 0 <= i < df.shape[0]]
        valid_D = D[0][:len(valid_indices)]
        results_df = df.iloc[valid_indices].copy()
        results_df['distance'] = valid_D

    # ----- Global Price Ordering -----
    if any(word in user_query.lower() for word in ["cheapest", "lowest price", "affordable", "low budget"]) or ("PriceOrder" in constraints and constraints["PriceOrder"] == "asc"):
        results_df = results_df.sort_values(by='price', ascending=True)
        return results_df.head(5).reset_index(drop=True)
    elif any(word in user_query.lower() for word in ["most expensive", "highest price", "priciest", "expensive", "high budget"]) or ("PriceOrder" in constraints and constraints["PriceOrder"] == "desc"):
        results_df = results_df.sort_values(by='price', ascending=False)
        return results_df.head(5).reset_index(drop=True)

    # ----- Additional Sorting by Carat if Explicit Keywords Present -----
    if any(word in user_query.lower() for word in ["highest", "largest", "maximum"]):
        results_df = results_df.sort_values(by='carat', ascending=False)
        return results_df.head(5).reset_index(drop=True)
    elif any(word in user_query.lower() for word in ["minimum", "lowest", "smallest"]):
        results_df = results_df.sort_values(by='carat', ascending=True)
        return results_df.head(5).reset_index(drop=True)
    else:
        def compute_score(row, constraints, df_filtered):
            score = row['distance']
            if "Carat" in constraints:
                score += 1000 * abs(row["carat"] - constraints["Carat"])
            else:
                median_carat = df_filtered['carat'].median()
                score += 100 * abs(row["carat"] - median_carat)
            if "Budget" in constraints:
                user_budget = constraints["Budget"]
                score += 0.05 * abs(float(row["price"]) - user_budget)
            else:
                try:
                    price = float(row["price"])
                except:
                    price = 0
                score += 0.1 * price
            if "Clarity" in constraints and row["clarity"].lower() != constraints["Clarity"].lower():
                score += 50
            if "Color" in constraints:
                if not row["color"].lower().startswith(constraints["Color"].lower()):
                    score += 50
            for attr, penalty in [("Cut", 20), ("Symmetry", 20), ("Polish", 20)]:
                if attr in constraints and row[attr.lower()].lower() != constraints[attr].lower():
                    score += penalty
            return score
        results_df['score'] = results_df.apply(lambda row: compute_score(row, constraints, df), axis=1)
        results_df = results_df.sort_values(by='score', ascending=True)
        return results_df.head(5).reset_index(drop=True)

# ------------------- Groq Integration -------------------
def generate_groq_response(user_query, relevant_data, client):
    prompt = f"""
You are a friendly and knowledgeable shop assistant at a diamond store.
Your goal is to help the customer find diamonds that best match their query.

Please analyze the following diamond details and produce a JSON response that includes the top matching diamonds.
Your response should include:
1. A brief introductory paragraph (one or two sentences) explaining what you found.
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

For example, your response should look like:
"I found several diamonds matching your criteria. Here are the best options:
<diamond-data>
[
  {{
    \\"Carat\\": 1.01,
    \\"Clarity\\": \\"vs1\\",
    \\"Color\\": \\"f\\",
    \\"Cut\\": \\"ex\\",
    \\"Shape\\": \\"round\\",
    \\"Price\\": \\"5000\\",
    \\"Style\\": \\"labgrown\\",
    \\"Polish\\": \\"ex\\",
    \\"Symmetry\\": \\"ex\\",
    \\"Lab\\": \\"igi\\",
    \\"Flo\\": \\"non\\"
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

# ------------------- Diamond Chatbot Logic -------------------
def diamond_chatbot(user_query, df, faiss_index, model, client):
    if user_query.strip().lower() in ["hi", "hello"]:
        return "Hey there! I'm your diamond guru 😎. Ready to help you find that perfect sparkle? Tell me what you're looking for!"

    constraints = extract_constraints_from_query(user_query)
    if not constraints and not any(keyword in user_query.lower() for keyword in ["maximum", "minimum", "lowest", "highest", "largest", "smallest"]):
        return "Hello! I'm your diamond assistant. Please let me know your preferred carat, clarity, color, cut, or budget so I can help you find the perfect diamond."

    results_df = hybrid_search(user_query, df, faiss_index, model, top_k=200)
    if results_df.empty:
        return "No matching diamonds found. Please try a different query."

    top_5 = results_df.head(5)
    relevant_data = "\n".join(top_5['combined_text'].tolist())
    groq_response = generate_groq_response(user_query, relevant_data, client)
    return groq_response

# ------------------- Markdown to HTML Conversion -------------------
def convert_markdown_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# ------------------- Main Execution -------------------
def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq()

    try:
        df, embeddings, index, model = load_data_and_index(EMBEDDING_FILE_PATH, FAISS_INDEX_FILE_PATH, DIAMONDS_CSV_FILE, MODEL_PATH)
        print("Data, embeddings, FAISS index, and model loaded from disk.")
    except Exception as e:
        print("Error loading existing data:", e)
        return

    while True:
        user_query = input("Hi! How can I help you? : ")
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for visiting! Have a wonderful day.")
            break

        # Optionally prompt for style if not specified
        constraints = extract_constraints_from_query(user_query)
        if "Style" not in constraints:
            style_input = input("Please specify the style (LabGrown or Natural): ")
            user_query += " " + style_input

        response = diamond_chatbot(user_query, df, index, model, client)
        print(response)
        print("\n---\n")

if __name__ == "__main__":
    main()
