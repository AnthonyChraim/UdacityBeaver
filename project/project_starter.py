import pandas as pd
import numpy as np
import os
import time
import ast
import re
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Tuple
from sqlalchemy import create_engine, Engine

try:
    import dotenv
except Exception:
    dotenv = None

try:
    # Optional dependency to align with a tool-oriented smolagents style.
    from smolagents import tool
except Exception:
    def tool(func):
        return func

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


if dotenv is not None:
    dotenv.load_dotenv()

ITEM_PRICE = {item["item_name"]: item["unit_price"] for item in paper_supplies}

ITEM_ALIASES = {
    "A4 paper": ["a4 paper", "a4 printer paper", "a4 printing paper", "a4 white paper", "a4 copier paper"],
    "Standard copy paper": ["standard copy paper", "standard printer paper", "printer paper", "copy paper"],
    "Cardstock": ["cardstock", "heavy cardstock", "white cardstock", "colored cardstock", "card stock"],
    "Colored paper": ["colored paper", "colorful paper", "bright-colored paper"],
    "Glossy paper": ["glossy paper", "glossy a4 paper"],
    "Matte paper": ["matte paper"],
    "Recycled paper": ["recycled paper", "eco-friendly paper"],
    "Poster paper": ["poster paper", "poster board", "posters", "large poster paper"],
    "Banner paper": ["banner paper", "banners"],
    "Construction paper": ["construction paper"],
    "Wrapping paper": ["wrapping paper", "decorative wrapping paper", "patterned wrapping paper"],
    "Letter-sized paper": ["letter-sized paper", "letter size paper"],
    "Letterhead paper": ["letterhead", "letterhead paper"],
    "Envelopes": ["envelopes", "envelope"],
    "Sticky notes": ["sticky notes"],
    "Notepads": ["notepads"],
    "Invitation cards": ["invitation cards", "tickets"],
    "Flyers": ["flyers"],
    "Party streamers": ["streamers", "party streamers"],
    "Decorative adhesive tape (washi tape)": ["washi tape", "decorative adhesive tape", "masking tape"],
    "Presentation folders": ["presentation folders", "folders"],
    "Paper plates": ["paper plates", "plates"],
    "Paper cups": ["paper cups", "cups"],
    "Paper napkins": ["paper napkins", "napkins", "table napkins"],
    "Table covers": ["table covers"],
    "Large poster paper (24x36 inches)": ["24x36 inches", "24x36", "large-format poster paper"],
    "Rolls of banner paper (36-inch width)": ["36-inch width", "rolls of banner paper"],
    "100 lb cover stock": ["100 lb cover stock"],
    "80 lb text paper": ["80 lb text paper"],
    "250 gsm cardstock": ["250 gsm cardstock", "250 gsm"],
    "220 gsm poster paper": ["220 gsm poster paper"],
}

UNIT_MULTIPLIERS = {
    "ream": 500,
    "reams": 500,
    "box": 250,
    "boxes": 250,
    "pack": 100,
    "packs": 100,
    "packet": 100,
    "packets": 100,
}


def _parse_dates(request_text: str) -> Tuple[str, str]:
    request_date_match = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", request_text)
    request_date = request_date_match.group(1) if request_date_match else datetime.now().strftime("%Y-%m-%d")

    due_match = re.search(r"by\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", request_text, flags=re.IGNORECASE)
    due_date = request_date
    if due_match:
        try:
            due_date = datetime.strptime(due_match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            due_date = request_date
    return request_date, due_date


def _extract_line_items(request_text: str) -> Tuple[Dict[str, int], List[str]]:
    text_lower = request_text.lower()
    matched: Dict[str, int] = {}
    unknown_mentions: List[str] = []

    generic_mentions = re.findall(
        r"(\d{1,3}(?:,\d{3})*)\s*(sheets?|reams?|rolls?|boxes?|packs?|packets?)?\s*(?:of\s+)?([a-z0-9\"'().\- ]{3,55})",
        text_lower,
    )

    for qty_raw, unit_raw, phrase in generic_mentions:
        if any(skip in phrase for skip in ["2025", "delivery", "april", "may"]):
            continue
        phrase = phrase.strip(" ,.-")
        if len(phrase) < 3:
            continue

        canonical_item = None
        for item_name, aliases in ITEM_ALIASES.items():
            if any(alias in phrase for alias in aliases):
                canonical_item = item_name
                break

        if canonical_item is None:
            if any(keyword in phrase for keyword in ["a3", "balloons", "cardboard", "tickets"]):
                unknown_mentions.append(phrase)
            continue

        qty = int(qty_raw.replace(",", ""))
        qty *= UNIT_MULTIPLIERS.get((unit_raw or "sheets").lower(), 1)
        matched[canonical_item] = matched.get(canonical_item, 0) + qty

    return matched, unknown_mentions


@tool
def inventory_snapshot_tool(as_of_date: str) -> Dict[str, int]:
    return get_all_inventory(as_of_date)


@tool
def stock_level_tool(item_name: str, as_of_date: str) -> int:
    stock_df = get_stock_level(item_name, as_of_date)
    return int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0


@tool
def supplier_eta_tool(request_date: str, quantity: int) -> str:
    return get_supplier_delivery_date(request_date, quantity)


@tool
def cash_balance_tool(as_of_date: str) -> float:
    return get_cash_balance(as_of_date)


@tool
def quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict]:
    return search_quote_history(search_terms, limit=limit)


@tool
def create_transaction_tool(item_name: str, transaction_type: str, quantity: int, price: float, date: str) -> int:
    return create_transaction(item_name, transaction_type, quantity, price, date)


@tool
def financial_report_tool(as_of_date: str) -> Dict:
    return generate_financial_report(as_of_date)


class InventoryAgent:
    def assess_availability(self, requested_items: Dict[str, int], request_date: str, due_date: str) -> Dict:
        reorder_plan: List[Dict] = []
        reorder_cost = 0.0

        for item_name, quantity in requested_items.items():
            current_stock = stock_level_tool(item_name, request_date)
            needed = max(0, quantity - current_stock)
            if needed <= 0:
                continue

            delivery_date = supplier_eta_tool(request_date, needed)
            if delivery_date > due_date:
                return {
                    "can_fulfill": False,
                    "reason": (
                        f"Cannot restock {item_name} in time. Earliest supplier date is {delivery_date}, "
                        f"after required date {due_date}."
                    ),
                    "reorder_plan": [],
                    "reorder_cost": 0.0,
                }

            cost = needed * ITEM_PRICE[item_name]
            reorder_cost += cost
            reorder_plan.append({"item_name": item_name, "quantity": needed, "cost": cost})

        available_cash = cash_balance_tool(request_date)
        if reorder_cost > available_cash * 0.9:
            return {
                "can_fulfill": False,
                "reason": f"Insufficient cash for reorder. Need ${reorder_cost:.2f}, available ${available_cash:.2f}.",
                "reorder_plan": [],
                "reorder_cost": reorder_cost,
            }

        return {
            "can_fulfill": True,
            "reason": "Inventory can satisfy request with feasible restock.",
            "reorder_plan": reorder_plan,
            "reorder_cost": reorder_cost,
        }

    def execute_reorder(self, reorder_plan: List[Dict], request_date: str) -> None:
        for reorder in reorder_plan:
            create_transaction_tool(
                item_name=reorder["item_name"],
                transaction_type="stock_orders",
                quantity=reorder["quantity"],
                price=float(reorder["cost"]),
                date=request_date,
            )


class QuotingAgent:
    def generate_quote(self, requested_items: Dict[str, int], job: str, event: str, need_size: str) -> Dict:
        subtotal = sum(ITEM_PRICE[item_name] * qty for item_name, qty in requested_items.items())
        total_units = sum(requested_items.values())

        historical_quotes = quote_history_tool([job, event, need_size], limit=5)
        size_discount = {"small": 0.02, "medium": 0.05, "large": 0.08}.get(need_size.lower(), 0.02)
        volume_discount = 0.05 if total_units >= 5000 else 0.03 if total_units >= 1500 else 0.0
        history_discount = 0.0

        if historical_quotes:
            average_historical = float(np.mean([float(q["total_amount"]) for q in historical_quotes]))
            if subtotal > average_historical * 1.5:
                history_discount = 0.03

        total_discount = min(0.2, size_discount + volume_discount + history_discount)
        total_amount = round(subtotal * (1 - total_discount), 2)

        explanation = (
            f"Catalog subtotal is ${subtotal:.2f}. Applied {total_discount*100:.1f}% total discount "
            f"(size {size_discount*100:.1f}%, volume {volume_discount*100:.1f}%, history {history_discount*100:.1f}%). "
            f"Final quote is ${total_amount:.2f}."
        )

        return {"subtotal": subtotal, "total_amount": total_amount, "explanation": explanation}


class SalesAgent:
    def finalize_sale(self, requested_items: Dict[str, int], quote_total: float, request_date: str) -> None:
        subtotal = sum(ITEM_PRICE[item_name] * qty for item_name, qty in requested_items.items())
        if subtotal <= 0:
            return
        ratio = quote_total / subtotal
        for item_name, quantity in requested_items.items():
            line_total = round(ITEM_PRICE[item_name] * quantity * ratio, 2)
            create_transaction_tool(
                item_name=item_name,
                transaction_type="sales",
                quantity=quantity,
                price=float(line_total),
                date=request_date,
            )

    def post_sale_summary(self, request_date: str) -> Dict:
        return financial_report_tool(request_date)


class OrchestratorAgent:
    def __init__(self):
        self.inventory_agent = InventoryAgent()
        self.quoting_agent = QuotingAgent()
        self.sales_agent = SalesAgent()

    def handle_request(self, request_text: str, job: str, event: str, need_size: str) -> str:
        request_date, due_date = _parse_dates(request_text)
        requested_items, unknown_mentions = _extract_line_items(request_text)

        if not requested_items:
            return (
                "Unable to generate a quote because no supported catalog items were parsed. "
                "Please provide item names from the catalog."
            )

        if unknown_mentions:
            return (
                "We cannot fulfill the full request because some items are outside our catalog: "
                + ", ".join(sorted(set(unknown_mentions))[:3])
                + "."
            )

        if sum(requested_items.values()) > 20000:
            return (
                "We cannot fulfill this request as submitted because the order volume exceeds "
                "our current single-order operational threshold."
            )

        assessment = self.inventory_agent.assess_availability(requested_items, request_date, due_date)
        if not assessment["can_fulfill"]:
            return f"Order cannot be fulfilled: {assessment['reason']}"

        quote = self.quoting_agent.generate_quote(requested_items, job, event, need_size)
        self.inventory_agent.execute_reorder(assessment["reorder_plan"], request_date)
        self.sales_agent.finalize_sale(requested_items, float(quote["total_amount"]), request_date)
        financials = self.sales_agent.post_sale_summary(request_date)

        return (
            f"Quote approved at ${quote['total_amount']:.2f}. {quote['explanation']} "
            f"Order is scheduled for delivery by {due_date}. "
            f"Updated cash balance is ${financials['cash_balance']:.2f}."
        )

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    orchestrator = OrchestratorAgent()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        response = orchestrator.handle_request(
            request_text=request_with_date,
            job=row["job"],
            event=row["event"],
            need_size=row["need_size"],
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
