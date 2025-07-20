import os
import json
import ccxt
import mysql.connector
import streamlit as st
from decimal import Decimal

###############################################################################
# 1) MySQL for meta_parameters
###############################################################################
def get_db_connection():
    host = os.getenv("DB_HOST", "localhost")
    user = os.getenv("DB_USER", "avarus_user")
    password = os.getenv("DB_PASS", "someStrongPassword")
    database = os.getenv("DB_NAME", "avarus2")
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

def load_latest_params():
    """
    Return only the newest param rows from meta_parameters
    (all rows sharing max(last_updated)).
    """
    conn = get_db_connection()
    c = conn.cursor()
    q_latest = "SELECT MAX(last_updated) FROM meta_parameters"
    c.execute(q_latest)
    row = c.fetchone()
    newest_ts = row[0] if row else None
    if not newest_ts:
        c.close()
        conn.close()
        return []

    q_params = """
      SELECT param_name, param_value, last_updated
      FROM meta_parameters
      WHERE last_updated = %s
      ORDER BY param_name
    """
    c.execute(q_params, (newest_ts,))
    rows = c.fetchall()
    c.close()
    conn.close()

    results = []
    for r in rows:
        pname = r[0]
        pval  = r[1]
        lup   = str(r[2]) if r[2] else ""
        try:
            parsed_val = json.loads(pval)
        except:
            parsed_val = pval
        results.append({
            "Name": pname,
            "Value": parsed_val,
            "LastUpdated": lup
        })
    return results

###############################################################################
# 2) Kraken read-only + caching
###############################################################################
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY","mLwzCnGTVDPNuU3MJStKw3TeNNEBgcUO/SDmQRm96icVFTwE/g4LQaWN")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET","ww+o/YsBtVRH2h58LjrfTTYtZBdaYBnwHSFVZbp/KJZiWrgFm+9NyNNfMaupRa3VyIQMw0uLm+xSFUe3kYvNHg==")

def init_kraken():
    return ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True
    })

@st.cache_data(ttl=60)
def fetch_spot_balances():
    """Fetch spot balances from Kraken (cached 60s)."""
    kr = init_kraken()
    return kr.fetch_balance(params={"type": "spot"})

@st.cache_data(ttl=60)
def fetch_open_orders():
    """Fetch open orders (cached 60s)."""
    kr = init_kraken()
    return kr.fetch_open_orders()

@st.cache_data(ttl=60)
def fetch_ticker(pair):
    """Fetch a ticker from Kraken (cached)."""
    kr = init_kraken()
    return kr.fetch_ticker(pair)

###############################################################################
# 3) Summation Helper (including USD)
###############################################################################
def sum_spot_in_eur(spot_bal):
    """
    Sums *all coins* in spot to EUR. If coin is USD, we convert via EUR/USD.
    If coin is e.g. BTC => we try BTC/EUR, fallback BTC/USD => then convert.
    """
    totals = spot_bal.get("total", {})
    total_eur = Decimal("0")

    # We'll cache the EUR/USD ticker once
    eurusd = Decimal("1")
    try:
        t_ = fetch_ticker("EUR/USD")
        last_ = t_.get("last") or t_.get("close") or 0
        eurusd = Decimal(str(last_)) if last_ else Decimal("1")
    except:
        pass

    for coin, qty in totals.items():
        if not qty or qty <= 0:
            continue
        if coin.startswith("."):
            continue

        qty_dec = Decimal(str(qty))
        if coin.upper() == "EUR":
            total_eur += qty_dec
        elif coin.upper() == "USD":
            # Directly convert to EUR => USD_value / (EUR/USD)
            val_eur = qty_dec / eurusd
            total_eur += val_eur
        else:
            # e.g. BTC, ETH => try coin/EUR
            px_eur = Decimal("0")
            try:
                ticker_ce = fetch_ticker(f"{coin}/EUR")
                ce_last = ticker_ce.get("last") or ticker_ce.get("close") or 0
                px_eur = Decimal(str(ce_last))
            except:
                # fallback coin/USD => then USD->EUR
                try:
                    ticker_cu = fetch_ticker(f"{coin}/USD")
                    cu_last = ticker_cu.get("last") or ticker_cu.get("close") or 0
                    px_usd = Decimal(str(cu_last))
                    if px_usd>0 and eurusd>0:
                        px_eur = px_usd / eurusd
                except:
                    px_eur = Decimal("0")

            total_eur += qty_dec * px_eur
    return total_eur

###############################################################################
# 4) Investors
###############################################################################
from decimal import Decimal
INVESTORS = {
    "Mark": Decimal("2500"),
    "Tijmen": Decimal("1500"),
    "Murrie": Decimal("1500"),
}
TOTAL_INITIAL = sum(INVESTORS.values())  # 5500

###############################################################################
# 5) Single password
###############################################################################
DASHBOARD_PASSWORD = "avarus68!"

def check_password():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        pw = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if pw == DASHBOARD_PASSWORD:
                st.session_state["logged_in"] = True
            else:
                st.error("Wrong password.")
                st.stop()
    return st.session_state["logged_in"]

###############################################################################
# 6) Force a rerun function for the first-click fix
###############################################################################
def force_rerun():
    """Try st.rerun if available, else st.stop() as fallback."""
    import streamlit
    # st.rerun was introduced in Streamlit 1.22+.
    if hasattr(streamlit, "rerun"):
        streamlit.rerun()
    else:
        st.stop()

###############################################################################
# 7) Navigation: 4 Buttons Below the Centered Logo
###############################################################################
def show_navbar():
    """
    4 columns => 4 buttons: Dashboard, Holdings, Open Orders, Parameters
    If clicked => update st.session_state["page"], then force rerun.
    """
    nav_cols = st.columns([1,1,1,1], gap="small")
    if nav_cols[0].button("Dashboard"):
        st.session_state["page"] = "Dashboard"
        force_rerun()
    if nav_cols[1].button("Holdings"):
        st.session_state["page"] = "Holdings"
        force_rerun()
    if nav_cols[2].button("Open Orders"):
        st.session_state["page"] = "OpenOrders"
        force_rerun()
    if nav_cols[3].button("Parameters"):
        st.session_state["page"] = "Parameters"
        force_rerun()

###############################################################################
# 8) Pages
###############################################################################
def page_dashboard():
    st.header("Avarus Dashboard")

    # fetch spot => sum in EUR (including USD => EUR)
    total_eur = Decimal("0")
    try:
        bal = fetch_spot_balances()
        total_eur = sum_spot_in_eur(bal)
    except Exception as e:
        st.warning(f"Error fetching spot => {e}")

    st.subheader("Total Portfolio Value")
    st.markdown(f"## € {total_eur:,.2f}")

    # Investors => no decimals
    st.subheader("Investor Breakdown")
    data = []
    if total_eur > 0:
        for name, amt in INVESTORS.items():
            fraction = amt / TOTAL_INITIAL
            net_val = fraction * total_eur
            data.append({
                "Investor": name,
                "Invested(€)": int(amt),
                "Portfolio%": f"{int(fraction*100)}%",
                "Current Net Value": int(net_val),
            })
    else:
        for name, amt in INVESTORS.items():
            data.append({
                "Investor": name,
                "Invested(€)": int(amt),
                "Portfolio%": "0%",
                "Current Net Value": 0,
            })
    st.table(data)

def page_holdings():
    st.header("Coin Holdings")

    try:
        bal = fetch_spot_balances()
        totals = bal.get("total", {})
        show_list = []
        for coin, qty in totals.items():
            if qty and qty >=0 and not coin.startswith("."):
                show_list.append((coin, qty))
        if show_list:
            st.table(show_list)
        else:
            st.info("No coin with quantity ≥1 found.")
    except Exception as e:
        st.error(f"Error => {e}")

def page_open_orders():
    st.header("Open Orders (Kraken)")

    try:
        od = fetch_open_orders()
        if od:
            table_data = []
            for o in od:
                table_data.append({
                    "OrderID": o.get("id"),
                    "Symbol": o.get("symbol"),
                    "Side": o.get("side"),
                    "Price": o.get("price"),
                    "Amount": o.get("amount"),
                    "Filled": o.get("filled"),
                    "Remaining": o.get("remaining"),
                    "Status": o.get("status"),
                    "DateTime": o.get("datetime"),
                })
            st.table(table_data)
        else:
            st.info("No open orders found.")
    except Exception as e:
        st.error(f"Error => {e}")

def page_parameters():
    st.header("Current ML Parameters (Most Recent)")

    pm = load_latest_params()
    if pm:
        st.table(pm)
    else:
        st.info("No parameters found or none updated recently.")

###############################################################################
# 9) Main
###############################################################################
def main():
    st.set_page_config(
        page_title="Avarus Dashboard",
        page_icon=":moneybag:",
        layout="wide",     # more mobile-friendly
        initial_sidebar_state="collapsed"
    )

    # 1) Center the logo
    top_cols = st.columns([1,2,1])
    with top_cols[1]:
        st.image("avaruslogo.webp", width=100)

    # 2) Check password
    if not check_password():
        return

    # 3) If no page => Dashboard
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    # 4) Show 4 nav buttons in a row
    show_navbar()

    # 5) Check which page to show
    page = st.session_state["page"]
    if page == "Dashboard":
        page_dashboard()
    elif page == "Holdings":
        page_holdings()
    elif page == "OpenOrders":
        page_open_orders()
    elif page == "Parameters":
        page_parameters()
    else:
        page_dashboard()

if __name__=="__main__":
    main()
