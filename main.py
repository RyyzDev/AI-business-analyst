from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv
import shutil
from datetime import datetime, time
import traceback
import gc
import google.generativeai as genai

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Retail Business Intelligence API",
    description="Stateless API for retail data analysis with AI-powered insights",
    version="1.0.0"
)

# ============================================================================
# CORS
# ============================================================================
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=true,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LLMs Init
# ============================================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("AI API Key tidak ditemukan! Pastikan file .env sudah benar.")

genai.configure(api_key=api_key)
model_ai = genai.GenerativeModel('gemini-2.5-flash')

# ============================================================================
# PYDANTIC MODELS FOR RESPONSE VALIDATION
# ============================================================================

class DataInsights(BaseModel):
    """Structured data insights from transaction analysis"""
    total_transactions: int
    total_unique_customers: int
    average_items_per_basket: float
    top_5_products: List[Dict[str, Any]]
    losers_5_products: List[Dict[str, Any]]
    peak_shopping_hours: List[Dict[str, Any]]
    shopping_by_part_of_day: Dict[str, int]
    date_range: Dict[str, str]

class AIRecommendations(BaseModel):
    """AI-generated strategic recommendations"""
    strategic_recommendations: str
    bundling_recommendations: str
    losers_discount_recommendations: str

class AnalysisResponse(BaseModel):
    """Complete API response model"""
    data_insights: DataInsights
    ai_recommendations: AIRecommendations
    status: str

# ============================================================================
# AI CONSULTANT FUNCTION
# ============================================================================

# def call_ai_consultant(data_summary: str) -> Dict[str, str]:
#     try:
#         # 1. Strategic Recommendations
#         strategic_response = model_ai.generate_content(
#             contents=data_summary + "\n\nBuatkan rekomendasi strategis."
#         )
        
#         # 2. Bundling Recommendations
#         bundling_response = model_ai.generate_content(
#             contents=data_summary + "\n\nBuatkan bundling strategis."
#         )
        
#         # 3. Losers Discount Recommendations
#         discount_response = model_ai.generate_content(
#             contents=data_summary + "\n\nBuatkan diskon strategis untuk losers produk."
#         )

#         def extract_text(resp):
#             if resp and hasattr(resp, 'text') and resp.text:
#                 return resp.text
#             return "No recommendations available from AI at this moment."

#         return {
#             "strategic_recommendations": extract_text(strategic_response),
#             "bundling_recommendations": extract_text(bundling_response),
#             "losers_discount_recommendations": extract_text(discount_response)
#         }

#     except Exception as e:
#         print(f"❌ Critical AI Error: {str(e)}")
#         return {
#             "strategic_recommendations": f"Error: {str(e)}",
#             "bundling_recommendations": "Service unavailable",
#             "losers_discount_recommendations": "Service unavailable"
#         }

    # MOCK RESPONSE - Replace with real API call
    return {
        "strategic_recommendations": """#**CROSS-SELLING OPPORTUNITIES:**

Based on the shopping behavior analysis, here are key cross-selling strategies:

1. **Bundle Premium Products**: The top-selling items show strong customer preference. Create bundle packages combining these popular products with complementary items to increase basket size and average transaction value.

2. **Time-Based Product Pairing**: Analysis reveals distinct shopping patterns throughout the day. Morning shoppers tend to purchase different product categories than evening shoppers. Develop targeted product recommendations based on time-of-day patterns to maximize cross-sell conversion.

3. **Customer Segmentation Strategy**: With varied average basket sizes, implement tiered cross-selling approaches - suggest premium add-ons to high-value customers while offering value bundles to price-sensitive segments.

**PROMOTION SCHEDULE RECOMMENDATIONS:**

1. **Peak Hour Flash Sales**: Deploy flash promotions during identified peak shopping hours to capitalize on high traffic periods. This maximizes exposure and creates urgency among the highest volume of customers.

2. **Off-Peak Incentives**: Introduce special discounts during slower periods (identified in the part-of-day analysis) to redistribute customer traffic and optimize resource utilization throughout the day.

3. **Morning Coffee + Product Bundles**: If morning shows significant traffic, create breakfast/morning-time bundles. For evening peaks, focus on dinner-related or convenience product combinations.

4. **Weekend vs Weekday Strategy**: Analyze the date patterns to identify weekly trends and schedule major promotions on days with historically highest conversion rates.

**ACTIONABLE NEXT STEPS:**
- Implement A/B testing for bundle recommendations during peak hours
- Track cross-sell conversion rates by time period
- Monitor basket size changes after implementing time-based recommendations
- Use product co-occurrence data to refine bundle offerings monthly""",

        "bundling_recommendations": """**STRATEGIC BUNDLING RECOMMENDATIONS:**

1. **Winner-Winner Bundles (Premium Tier)**
   - Pair top 2-3 best-selling products together
   - Price at 10-15% discount vs individual purchase
   - Target: High-value customers during peak hours
   - Example: "Paket Favorit Pelanggan" featuring your most popular items
   - Expected Impact: Increase average transaction value by 20-30%

2. **Winner-Loser Bundles (Value Tier)**
   - Combine 1 best-seller with 1-2 slow-moving products
   - Offer 20-25% discount to incentivize trial of losers products
   - Target: Price-sensitive customers and new customer acquisition
   - Example: "Paket Hemat Spesial" - 1 produk populer + 2 produk baru
   - Expected Impact: Clear 30-40% of loser inventory while maintaining margin

3. **Time-Based Combo Packs**
   - Morning Bundle: Top breakfast items (07:00-11:00)
   - Lunch Bundle: Popular midday products (11:00-14:00)
   - Evening Bundle: Dinner/snack combinations (17:00-21:00)
   - Price: 15% discount during respective time slots only
   - Expected Impact: Smooth demand curves and reduce operational bottlenecks

4. **Quantity-Based Bundling**
   - "Beli 3 Gratis 1" for medium-performing products
   - "Paket Keluarga" (Family Pack) with mixed product categories
   - Encourage bulk purchases to increase basket size
   - Expected Impact: 25% increase in units per transaction

5. **Complementary Product Pairing**
   - Analyze purchase patterns to identify products bought together
   - Create natural pairings (e.g., Kopi + Roti, Nasi + Lauk)
   - Display bundles prominently at checkout or high-traffic areas
   - Expected Impact: Impulse purchases increase by 15-20%

**IMPLEMENTATION TIMELINE:**
- Week 1-2: Launch Winner-Loser bundles to move slow inventory
- Week 3-4: Introduce Time-Based combos after measuring initial response
- Month 2: Roll out Premium Winner-Winner bundles based on refined data
- Ongoing: A/B test pricing and combinations monthly""",

        "losers_discount_recommendations": """**DISCOUNT STRATEGIES FOR SLOW-MOVING PRODUCTS:**

**IMMEDIATE ACTIONS (Week 1-2):**

1. **Aggressive Clearance Pricing (Bottom 5 Products)**
   - Apply 30-50% discount on the 5 slowest-moving items
   - Display with "Diskon Spesial Minggu Ini!" signage
   - Create urgency with "Stok Terbatas" messaging
   - Time-Limited: Offer valid for 2 weeks maximum
   - Goal: Clear 60-70% of existing inventory

2. **BOGO (Buy One Get One) Promotions**
   - Identify losers products with highest profit margins
   - Offer "Beli 1 Gratis 1" for same product or "Beli 1 + 1 Gratis Produk Pilihan"
   - Cross-promote losers with complementary items
   - Duration: 1 week flash promotion
   - Goal: Accelerate turnover without heavy margin loss

3. **Bundle Attachment Strategy**
   - Automatically include 1 loser product FREE with every purchase above certain threshold
   - Example: "Gratis [Loser Product] untuk pembelian di atas Rp 50,000"
   - Introduce customers to products they wouldn't normally try
   - Goal: Convert non-buyers into potential repeat customers

**SHORT-TERM TACTICS (Week 3-4):**

4. **Loyalty Program Integration**
   - Offer 2x loyalty points for purchasing losers products
   - Create "Mystery Box" promotions featuring losers at 40% off
   - Reward adventurous customers willing to try new items
   - Goal: Build positive associations with underperforming products

5. **Peak Hour Loss-Leader Strategy**
   - Sell losers at cost (or slight loss) during peak traffic times
   - Position near checkout as impulse add-ons
   - Accept thin margins to drive foot traffic and basket additions
   - Goal: Increase overall transaction volume

6. **Sampling & Trial Programs**
   - Offer free samples of losers products to customers
   - Pair with QR code for instant 25% discount coupon
   - Gather feedback to understand why products underperform
   - Goal: Convert hesitant customers through trial experience

**MID-TERM RESTRUCTURING (Month 2-3):**

7. **Product Repositioning**
   - Rebrand/repackage losers with new names or presentation
   - Change product placement to high-visibility areas
   - Retrain staff to actively recommend these items
   - Test price elasticity with gradual 10-20% reductions
   - Goal: Identify if poor sales are due to awareness vs appeal

8. **Customer Segmentation Targeting**
   - Analyze which customer segments (by time, demographics) might prefer losers
   - Send targeted SMS/email promotions: "Diskon Eksklusif 40% untuk [Product]"
   - Personalize offers based on past purchase behavior
   - Goal: Find niche audiences for underperforming products

9. **Last Resort: Discontinuation Analysis**
   - If products remain losers after 2 months of aggressive tactics
   - Calculate total loss from holding inventory vs liquidation
   - Consider donating to charity for tax benefits and goodwill
   - Replace with new products based on customer demand data
   - Goal: Optimize inventory mix for profitability

**MONITORING & SUCCESS METRICS:**
- Track weekly sales velocity for each loser product
- Measure conversion rate of discount campaigns
- Monitor profit margins to ensure sustainability
- Collect customer feedback on why they purchased (or didn't)
- Adjust strategies bi-weekly based on performance data

**EXPECTED OUTCOMES:**
- 50-70% reduction in loser product inventory within 60 days
- Discovery of 1-2 hidden gems that perform better with repositioning
- Improved overall inventory turnover ratio
- Data-driven insights for future product selection"""
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cleanup_temp_file(file_path: str) -> None:
    """
    Background task to securely delete temporary files.
    Ensures zero persistence on server.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Temp file deleted: {file_path}")
    except Exception as e:
        print(f"⚠ Warning: Failed to delete {file_path}: {str(e)}")

def validate_file_type(filename: str) -> str:
    """
    Validates uploaded file is CSV or Excel format.
    Returns normalized extension.
    """
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    ext = filename.lower().split('.')[-1]
    if ext not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: .{ext}. Only CSV and XLSX files are accepted."
        )
    return ext

def load_dataframe(file_path: str, extension: str) -> pd.DataFrame:
    """
    Loads file into Pandas DataFrame based on extension.
    Handles both CSV and Excel formats.
    """
    try:
        if extension == 'csv':
            df = pd.read_csv(file_path)
        else:  # xlsx or xls
            df = pd.read_excel(file_path)
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="File is empty or contains no valid data"
            )
        
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {str(e)}"
        )

def validate_mandatory_columns(df: pd.DataFrame) -> None:
    """
    Validates presence of all 5 mandatory columns.
    Returns HTTP 400 if any column is missing.
    """
    mandatory_columns = [
        'Nama_Pelanggan',
        'Tanggal_Pembelian',
        'Waktu_Pembelian',
        'Produk_Dibeli',
        'Jumlah_Item'
    ]
    
    missing = [col for col in mandatory_columns if col not in df.columns]
    
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing mandatory columns",
                "missing_columns": missing,
                "required_columns": mandatory_columns,
                "found_columns": df.columns.tolist()
            }
        )
    
    print(f"✓ All mandatory columns present: {mandatory_columns}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive data cleaning:
    1. Handle missing values
    2. Remove duplicate transactions
    3. Validate data types
    """
    initial_rows = len(df)
    
    # Handle missing values - drop rows with critical missing data
    critical_columns = ['Nama_Pelanggan', 'Tanggal_Pembelian', 'Produk_Dibeli']
    df = df.dropna(subset=critical_columns)
    print(f"✓ Dropped {initial_rows - len(df)} rows with missing critical data")
    
    # Remove duplicate transactions
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {initial_rows - len(df)} duplicate transactions")
    
    # Fill missing Jumlah_Item with 1 (assume single item if not specified)
    df['Jumlah_Item'] = df['Jumlah_Item'].fillna(1)
    
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid data remaining after cleaning process"
        )
    
    return df

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Tanggal_Pembelian to datetime objects.
    Handles various date formats.
    """
    try:
        df['Tanggal_Pembelian'] = pd.to_datetime(
            df['Tanggal_Pembelian'], 
            errors='coerce'
        )
        
        # Drop rows where date parsing failed
        invalid_dates = df['Tanggal_Pembelian'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠ Warning: {invalid_dates} invalid dates found and removed")
            df = df.dropna(subset=['Tanggal_Pembelian'])
        
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid dates found in Tanggal_Pembelian column"
            )
        
        print(f"✓ Date parsing complete: {len(df)} valid transactions")
        return df
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Date parsing failed: {str(e)}"
        )

def explode_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Splits comma-separated products into individual rows.
    This ensures accurate per-product counting and popularity analysis.
    
    Example transformation:
    Before: | Customer | Products: "Kopi, Roti, Susu" | Qty: 3 |
    After:  | Customer | Product: "Kopi" | Qty: 1 |
            | Customer | Product: "Roti" | Qty: 1 |
            | Customer | Product: "Susu" | Qty: 1 |
    """
    try:
        # Split products by comma and strip whitespace
        df['Produk_List'] = df['Produk_Dibeli'].str.split(',')
        df['Produk_List'] = df['Produk_List'].apply(
            lambda x: [item.strip() for item in x] if isinstance(x, list) else [str(x).strip()]
        )
        
        # Explode into individual product rows
        df_exploded = df.explode('Produk_List')
        
        # Clean up product names
        df_exploded['Produk_Individual'] = df_exploded['Produk_List'].str.strip()
        df_exploded = df_exploded[df_exploded['Produk_Individual'] != '']
        
        # Calculate items per product (divide total by number of products)
        df_exploded['Items_Per_Product'] = 1  # Each exploded row represents 1 product instance
        
        print(f"✓ Product explosion: {len(df)} → {len(df_exploded)} product rows")
        return df_exploded
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Product explosion failed: {str(e)}"
        )

def create_part_of_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering: Creates 'Part of Day' categorization.
    
    Categories:
    - Morning: 05:00 - 11:59
    - Afternoon: 12:00 - 17:59
    - Evening: 18:00 - 04:59
    """
    try:
        def categorize_time(waktu_str):
            """Categorizes time string into part of day"""
            try:
                # Handle various time formats
                if pd.isna(waktu_str):
                    return 'Unknown'
                
                # Try to parse time
                if isinstance(waktu_str, time):
                    hour = waktu_str.hour
                elif isinstance(waktu_str, str):
                    # Parse time string (e.g., "14:30" or "14:30:00")
                    time_obj = pd.to_datetime(waktu_str, format='%H:%M:%S', errors='coerce')
                    if pd.isna(time_obj):
                        time_obj = pd.to_datetime(waktu_str, format='%H:%M', errors='coerce')
                    
                    if pd.isna(time_obj):
                        return 'Unknown'
                    hour = time_obj.hour
                else:
                    return 'Unknown'
                
                # Categorize by hour
                if 5 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                else:
                    return 'Evening'
                    
            except Exception:
                return 'Unknown'
        
        df['Part_of_Day'] = df['Waktu_Pembelian'].apply(categorize_time)
        print(f"✓ Part of Day feature created: {df['Part_of_Day'].value_counts().to_dict()}")
        return df
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Part of Day feature creation failed: {str(e)}"
        )

def calculate_business_metrics(df: pd.DataFrame, df_exploded: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates comprehensive business intelligence metrics:
    1. Total transactions
    2. Average items per basket
    3. Top 5 most purchased products (WINNERS)
    4. Bottom 5 least purchased products (LOSERS)
    5. Peak shopping hours
    6. Shopping patterns by part of day
    """
    metrics = {}
    
    try:
        # 1. Total Transactions (from original dataset)
        metrics['total_transactions'] = int(len(df))
        metrics['total_unique_customers'] = int(df['Nama_Pelanggan'].nunique())
        
        # 2. Average Items Per Basket
        try:
            df['Jumlah_Item_Numeric'] = pd.to_numeric(df['Jumlah_Item'], errors='coerce').fillna(1)
            avg_items = float(df['Jumlah_Item_Numeric'].mean())
            metrics['average_items_per_basket'] = round(avg_items, 2)
        except Exception:
            metrics['average_items_per_basket'] = 1.0
        
        # 3. Top 5 Most Purchased Products - WINNERS (from exploded dataset)
        product_counts = df_exploded['Produk_Individual'].value_counts()
        top_5 = product_counts.head(5)
        metrics['top_5_products'] = [
            {
                'product_name': product,
                'purchase_count': int(count),
                'percentage': round((count / len(df_exploded)) * 100, 2)
            }
            for product, count in top_5.items()
        ]
        
        # 4. Bottom 5 Least Purchased Products - LOSERS (from exploded dataset)
        # Only include products with at least 2 occurrences to avoid one-off purchases
        losers_5 = product_counts[product_counts >= 2].tail(5).sort_values()
        metrics['losers_5_products'] = [
            {
                'product_name': product,
                'purchase_count': int(count),
                'percentage': round((count / len(df_exploded)) * 100, 2),
                'status': 'critical' if count < 5 else 'needs_attention'
            }
            for product, count in losers_5.items()
        ]
        
        # 5. Peak Shopping Hours
        df_with_hour = df.copy()
        try:
            df_with_hour['Hour'] = pd.to_datetime(
                df_with_hour['Waktu_Pembelian'], 
                format='%H:%M:%S', 
                errors='coerce'
            ).dt.hour
            
            # Fill NA hours with alternative parsing
            df_with_hour['Hour'] = df_with_hour['Hour'].fillna(
                pd.to_datetime(df_with_hour['Waktu_Pembelian'], format='%H:%M', errors='coerce').dt.hour
            )
            
            hour_counts = df_with_hour['Hour'].value_counts().head(5).sort_index()
            metrics['peak_shopping_hours'] = [
                {
                    'hour': f"{int(hour)}:00",
                    'transaction_count': int(count)
                }
                for hour, count in hour_counts.items() if not pd.isna(hour)
            ]
        except Exception:
            metrics['peak_shopping_hours'] = []
        
        # 6. Shopping by Part of Day
        part_of_day_counts = df_exploded['Part_of_Day'].value_counts().to_dict()
        metrics['shopping_by_part_of_day'] = {
            str(k): int(v) for k, v in part_of_day_counts.items()
        }
        
        # 7. Date Range
        metrics['date_range'] = {
            'start_date': df['Tanggal_Pembelian'].min().strftime('%Y-%m-%d'),
            'end_date': df['Tanggal_Pembelian'].max().strftime('%Y-%m-%d')
        }
        
        print(f"✓ Business metrics calculated successfully")
        print(f"  - Top products identified: {len(metrics['top_5_products'])}")
        print(f"  - Loser products identified: {len(metrics['losers_5_products'])}")
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics calculation failed: {str(e)}"
        )

def prepare_ai_summary(metrics: Dict[str, Any]) -> str:
    """
    Constructs a comprehensive data summary for AI analysis.
    Formats metrics into natural language for LLM processing.
    """
    summary = f"""**SUMMARY**

**TRANSACTION OVERVIEW:**
- Total Transaksi: {metrics['total_transactions']:,}
- Unique Customers: {metrics['total_unique_customers']:,}
- Rata-rata Items per Basket: {metrics['average_items_per_basket']}
- Analysis Period: {metrics['date_range']['start_date']} to {metrics['date_range']['end_date']}

**TOP 5 BEST-SELLING PRODUCTS:**
"""
    for i, product in enumerate(metrics['top_5_products'], 1):
        summary += f"{i}. {product['product_name']}: {product['purchase_count']:,} purchases ({product['percentage']}% of total)\n"
    
    summary += f"""
**BOTTOM 5 SLOW-MOVING PRODUCTS:**
"""
    for i, product in enumerate(metrics['losers_5_products'], 1):
        status_label = "🚨 CRITICAL" if product['status'] == 'critical' else "⚠️ NEEDS ATTENTION"
        summary += f"{i}. {product['product_name']}: {product['purchase_count']:,} purchases ({product['percentage']}%) - {status_label}\n"
    
    summary += f"""
**PEAK SHOPPING HOURS:**
"""
    for hour_data in metrics['peak_shopping_hours']:
        summary += f"- {hour_data['hour']}: {hour_data['transaction_count']:,} transactions\n"
    
    summary += f"""
**SHOPPING PATTERNS BY TIME OF DAY:**
"""
    for part, count in metrics['shopping_by_part_of_day'].items():
        summary += f"- {part}: {count:,} product purchases\n"
    
    summary += """
**ANALYSIS REQUEST:**
Based on this shopping behavior data:
1. Identifikasi peluang penjualan silang (produk yang sering dibeli bersama) dan sarankan jadwal promosi berdasarkan waktu belanja puncak.
2. Berikan strategi penggabungan yang menggabungkan produk terlaris dengan produk yang penjualannya kurang baik.
3. Buat strategi diskon dan obral khusus untuk produk-produk yang merugi guna mempercepat perputaran persediaan."""
    
    return summary

# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.post("/analyze-business", response_model=AnalysisResponse)
async def analyze_business(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Main endpoint for retail business intelligence analysis.
    
    WORKFLOW:
    1. Validate file type (CSV/XLSX)
    2. Check mandatory columns
    3. Clean data (missing values, duplicates)
    4. Parse datetime
    5. Explode products (comma-separated → individual rows)
    6. Create Part of Day feature
    7. Calculate business metrics
    8. Generate AI recommendations
    9. Return structured insights
    10. Clean up (background task)
    """
    temp_file_path = None
    df = None
    df_exploded = None
    
    try:
        # STEP 1: Validate File Type
        extension = validate_file_type(file.filename)
        print(f"✓ File type validated: {file.filename} (.{extension})")
        
        # STEP 2: Save to Temporary File
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        print(f"✓ Uploaded to temp: {temp_file_path}")
        
        # STEP 3: Load DataFrame
        df = load_dataframe(temp_file_path, extension)
        print(f"✓ DataFrame loaded: {len(df)} rows × {len(df.columns)} columns")
        
        # STEP 4: Validate Mandatory Columns
        validate_mandatory_columns(df)
        
        # STEP 5: Data Cleaning
        df = clean_data(df)
        
        # STEP 6: Parse Datetime
        df = parse_datetime(df)
        
        # STEP 7: Explode Products (CRITICAL for accurate counting)
        df_exploded = explode_products(df)
        
        # STEP 8: Create Part of Day Feature
        df_exploded = create_part_of_day_feature(df_exploded)
        
        # STEP 9: Calculate Business Metrics
        metrics = calculate_business_metrics(df, df_exploded)
        
        # STEP 10: Prepare AI Summary
        ai_summary = prepare_ai_summary(metrics)
        print(f"✓ AI summary prepared ({len(ai_summary)} chars)")
        
        # STEP 11: Get AI Recommendations (3 types)
        ai_recommendations = call_ai_consultant(ai_summary)
        print(f"✓ AI recommendations generated:")
        print(f"  - Strategic: {len(ai_recommendations['strategic_recommendations'])} chars")
        print(f"  - Bundling: {len(ai_recommendations['bundling_recommendations'])} chars")
        print(f"  - Losers Discount: {len(ai_recommendations['losers_discount_recommendations'])} chars")
        
        # STEP 12: Construct Response
        response_data = {
            "data_insights": metrics,
            "ai_recommendations": ai_recommendations,
            "status": "success"
        }
        
        # STEP 13: Schedule Cleanup
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # STEP 14: Explicit Memory Cleanup
        del df
        del df_exploded
        gc.collect()
        print(f"✓ Memory cleared")
        
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        # Re-raise HTTP exceptions with cleanup
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        if df is not None:
            del df
        if df_exploded is not None:
            del df_exploded
        gc.collect()
        raise
        
    except Exception as e:
        # Handle unexpected errors
        print(f"❌ ERROR: {traceback.format_exc()}")
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        if df is not None:
            del df
        if df_exploded is not None:
            del df_exploded
        gc.collect()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Retail Business Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/analyze-business (POST)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "operational",
        "memory_management": "stateless"
    }