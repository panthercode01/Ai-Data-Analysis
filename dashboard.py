
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="AI Sales Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Sales Data Analyst")
st.caption("Analyze sales data and ask questions about your CSV.")

# ---------- DATA SOURCE ----------
st.sidebar.header("Data Source")

source = st.sidebar.radio(
    "Choose data source",
    ["Default Sales Data", "Upload CSV", "CSV Link"]
)

df = None

if source == "Default Sales Data":
    try:
        df = pd.read_csv("sales.csv")
    except FileNotFoundError:
        st.error("sales.csv file nahi mili.")

elif source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("CSV file read nahi ho payi.")

else:
    csv_url = st.sidebar.text_input(
        "Direct HTTPS CSV URL",
        placeholder="https://example.com/data.csv"
    )
    if csv_url:
        if not csv_url.lower().startswith("https://"):
            st.warning("Security ke liye sirf HTTPS links allowed hain.")
        else:
            try:
                import requests
                response = requests.get(
                    csv_url,
                    timeout=10,
                    stream=True,
                    headers={"Range": "bytes=0-2000000"}
                )
                response.raise_for_status()

                content = response.raw.read(2_000_001)
                if len(content) > 2_000_000:
                    st.error("CSV 2 MB se chhoti honi chahiye.")
                else:
                    from io import BytesIO
                    df = pd.read_csv(BytesIO(content))

                response.close()
            except Exception:
                st.error("CSV link access nahi ho saka. Direct HTTPS CSV link try karo.")

# ---------- DATA VALIDATION ----------
if df is not None:
    if df.empty:
        st.warning("CSV file empty hai.")
        st.stop()

    if len(df) > 10000:
        st.warning("Maximum 10,000 rows allowed hain.")
        st.stop()

    df.columns = df.columns.str.strip().str.lower()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    # ---------- SALES ANALYSIS ----------
    required = {
        "product", "quantity", "unit_price", "unit_cost"
    }

    if required.issubset(df.columns):
        for col in ["quantity", "unit_price", "unit_cost"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[
            "product", "quantity", "unit_price", "unit_cost"
        ]).copy()

        df["revenue"] = df["quantity"] * df["unit_price"]
        df["profit"] = df["quantity"] * (
            df["unit_price"] - df["unit_cost"]
        )

        st.subheader("📈 Sales Dashboard")

        revenue = df["revenue"].sum()
        profit = df["profit"].sum()
        units = df["quantity"].sum()
        margin = profit / revenue * 100 if revenue else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"₹{revenue:,.2f}")
        c2.metric("Total Profit", f"₹{profit:,.2f}")
        c3.metric("Units Sold", f"{units:,.0f}")
        c4.metric("Profit Margin", f"{margin:.2f}%")

        product_sales = df.groupby("product", as_index=False).agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum")
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Revenue by Product")
            st.bar_chart(product_sales.set_index("product")["revenue"])

        with right:
            st.subheader("Profit by Product")
            st.bar_chart(product_sales.set_index("product")["profit"])

        st.subheader("Product Summary")
        st.dataframe(product_sales, use_container_width=True)

    else:
        st.info(
            "Ye CSV general analysis aur AI questions ke liye available hai. "
            "Sales metrics ke liye product, quantity, unit_price, "
            "unit_cost columns chahiye."
        )

    # ---------- AI CHAT ----------
    st.divider()
    st.subheader("💬 Ask AI About Your Data")

    question = st.text_input(
        "Ask a question",
        placeholder="Summarize this data"
    )

    if st.button("Ask AI"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            st.error("GEMINI_API_KEY .env file mein set nahi hai.")
        elif not question.strip():
            st.warning("Pehle question likho.")
        else:
            # Simple fixed questions are answered without API calls
            q = question.lower().strip()

            if "how many rows" in q or "row count" in q:
                st.write(f"Dataset mein {len(df)} rows hain.")

            elif "how many columns" in q or "column count" in q:
                st.write(f"Dataset mein {len(df.columns)} columns hain.")

            else:
                # Basic session-level request cap
                today = time.strftime("%Y-%m-%d")
                if st.session_state.get("api_day") != today:
                    st.session_state["api_day"] = today
                    st.session_state["api_calls"] = 0

                if st.session_state.get("api_calls", 0) >= 5:
                    st.error("Is session mein daily 5 AI requests ki limit reach ho gayi.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)

                        # Send only a small preview, not the entire CSV
                        preview = df.head(20).to_csv(index=False)
                        prompt = f"""
You are a concise sales/data analyst.
Answer based only on the provided CSV preview.
If the preview is insufficient, clearly say so.
Never invent values. Keep the answer under 100 words.

CSV columns: {list(df.columns)}
Total rows: {len(df)}
CSV preview (up to 20 rows):
{preview}

User question: {question}
"""

                        st.session_state["api_calls"] = (
                            st.session_state.get("api_calls", 0) + 1
                        )

                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                            config={"max_output_tokens": 150}
                        )
                        st.write(response.text or "No response returned.")

                    except Exception as e:
                        st.error(
                            "AI request failed. Check model availability, "
                            "API quota and key. Details: " + str(e)[:250]
                        )

else:
    st.info("Default data use karo, CSV upload karo, ya CSV link do.")

st.caption("Built with Python, Pandas, Streamlit and Gemini API.")
