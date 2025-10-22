import os, yaml
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Local modules ---
from data_fetcher import fetch_history
from model_trainer import load_ensemble, make_features, predict_proba_ensemble
from signal_engine import news_sentiment_score, combine_signal
from universe_fetcher import fetch_bist_all, fetch_sp500_wiki, fetch_nasdaq_all, get_cached_universes
from portfolio_backtest import backtest_multi
from binance_portfolio import build_returns_matrix, optimize_and_backtest

# ---------- Setup ----------
load_dotenv()
st.set_page_config(layout="wide", page_title="Mini‑Terminal Pro — ULTIMATE V9 (Patched)")

# Inject premium CSS if present
from pathlib import Path
css_path = Path("assets/style.css")
if css_path.exists():
    try:
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Stil dosyası okunamadı: {e}")
else:
    st.info("Not: assets/style.css bulunamadı. (Tema olmadan devam ediliyor.)")

st.markdown('<h1 style="margin-bottom:0.2rem;">Mini‑Terminal Pro — ULTIMATE V9</h1>', unsafe_allow_html=True)
st.caption("Koyu tema • Neon vurgu • Terminal görünümü")

# Load config
try:
    with open("config.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    st.error(f"config.yaml okunamadı: {e}")
    st.stop()

# ---------- Tabs ----------
tabs = st.tabs([
    "Evren Yönetimi",
    "Keşfet & Grafik",
    "Yapay Zeka Tahmin",
    "News Trading (Canlı)",
    "Portföy (Hisse)",
    "Kripto (Binance)"
])

# Helper: safe universe getter
def safe_universe():
    try:
        u = get_cached_universes()
        for k, v in u.items():
            if isinstance(v, pd.DataFrame):
                u[k] = v.drop_duplicates()
        return u
    except Exception as e:
        st.error(f"Evren yüklenemedi: {e}")
        return {"bist": pd.DataFrame(), "sp500": pd.DataFrame(), "nasdaq": pd.DataFrame()}

# -------- Evren Yönetimi --------
with tabs[0]:
    st.subheader("BIST + S&P500 + NASDAQ — Evren Güncelle")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("BIST: Hepsini Çek (Finnhub)", key="btn_bist"):
            try:
                df = fetch_bist_all(write_csv=True)
                st.success(f"BIST güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    with c2:
        if st.button("S&P 500: Wikipedia'dan Güncelle", key="btn_sp500"):
            try:
                df = fetch_sp500_wiki(write_csv=True)
                st.success(f"S&P 500 güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    with c3:
        if st.button("NASDAQ: Listeyi Güncelle", key="btn_nasdaq"):
            try:
                df = fetch_nasdaq_all(write_csv=True)
                st.success(f"NASDAQ güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    uni = safe_universe()
    st.write({k: (len(v) if isinstance(v, pd.DataFrame) else 0) for k,v in uni.items()})

# -------- Keşfet & Grafik --------
with tabs[1]:
    st.subheader("Sektör filtreli grafik")
    uni = safe_universe()
    src = st.selectbox("Kaynak", ["bist","sp500","nasdaq"], key="explore_src")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        sector_list = sorted([s for s in dfu['sector'].dropna().unique().tolist() if s])
        sector_sel = st.multiselect("Sektör filtresi (opsiyonel)", sector_list, key="explore_sector")
        dff = dfu if not sector_sel else dfu[dfu['sector'].isin(sector_sel)]
        options = dff['symbol'].dropna().astype(str).tolist()
        symbol = st.selectbox("Sembol", options, key="explore_symbol")
        if symbol:
            dfp = fetch_history(symbol, start=cfg['data']['start_date'])
            if dfp is None or dfp.empty:
                st.warning("Veri bulunamadı.")
            else:
                fig = go.Figure(data=[go.Candlestick(x=dfp.index, open=dfp['Open'], high=dfp['High'], low=dfp['Low'], close=dfp['Close'])])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dfp.tail(10))

# -------- Yapay Zeka Tahmin --------
with tabs[2]:
    st.subheader("YUKARI / AŞAĞI tahmini ve güven skoru")
    uni = safe_universe()
    src = st.selectbox("Kaynak (tahmin)", ["bist","sp500","nasdaq"], key="pred_src")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        options = dfu['symbol'].dropna().astype(str).tolist()
        symbol = st.selectbox("Sembol (tahmin)", options, key="pred_sym")
        models = load_ensemble()
        if not models:
            st.info("Model yok. `models/` klasörüne en az bir LightGBM .pkl koyun veya eğitim yapın.")
        else:
            df = fetch_history(symbol, start=cfg['data']['start_date'])
            if df is None or df.empty:
                st.warning("Veri yok.")
            else:
                feat = make_features(df)
                try:
                    X = feat[cfg['model']['features']].iloc[-1:]
                except Exception as e:
                    st.error(f"Özellikler hazırlanamadı: {e}")
                    X = None
                prob = predict_proba_ensemble(models, X) if X is not None else 0.5
                if prob is None:
                    prob = 0.5
                senti, _ = news_sentiment_score(symbol, hours=cfg['signal']['sentiment_lookback_days']*8)
                score = combine_signal(prob, senti, cfg)
                label = "YUKARI" if prob >= 0.5 else "AŞAĞI"
                st.metric("Tahmin", label, delta=f"Güven: {prob:.2f}")
                st.metric("Haber Sentiment", f"{senti:.3f}")
                st.metric("Kombine Skor", f"{score:.2f}")
                st.caption("Not: Demo amaçlıdır; yatırım tavsiyesi değildir.")

# -------- News Trading (Canlı) --------
with tabs[3]:
    st.subheader("Anlık Haber Akışı — Sentiment Momentum")
    st.markdown(f"<meta http-equiv='refresh' content='{cfg['signal']['news_refresh_seconds']}'>", unsafe_allow_html=True)
    uni = safe_universe()
    src = st.selectbox("Kaynak (haber)", ["bist","sp500","nasdaq"], key="news_src")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        options = dfu['symbol'].dropna().astype(str).tolist()
        symbol = st.selectbox("Sembol (haber)", options, key="news_sym")
        hours = st.slider("Son kaç saat", 1, 24, 6, key="news_hours")
        try:
            senti_now, news_df = news_sentiment_score(symbol, hours=hours)
            prev_senti, _ = news_sentiment_score(symbol, hours=min(hours*2,24))
        except Exception as e:
            st.error(f"Haber/duygu verisi alınamadı: {e}")
            senti_now, prev_senti, news_df = 0.0, 0.0, pd.DataFrame()
        trend = "↑ Artış" if senti_now > prev_senti else ("↓ Düşüş" if senti_now < prev_senti else "→ Durağan")
        st.metric("Anlık Haber Sentiment", f"{senti_now:.3f}", delta=trend)
        st.caption(f"Otomatik yenileme: {cfg['signal']['news_refresh_seconds']} sn")
        if news_df is None or news_df.empty:
            st.info("Bu pencerede haber bulunamadı.")
        else:
            try:
                st.dataframe(news_df[['datetime','headline','score']])
            except Exception:
                st.dataframe(news_df)

# -------- Portföy (Hisse) --------
with tabs[4]:
    st.subheader("Sektör bazlı portföy optimizasyonu + maliyetli backtest")
    uni = safe_universe()
    src = st.selectbox("Kaynak (portföy)", ["bist","sp500","nasdaq"], key="port_src")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        sector_list = sorted([s for s in dfu['sector'].dropna().unique().tolist() if s])
        sector_sel = st.multiselect("Sektör filtre (opsiyonel)", sector_list, key="port_sector")
        pool = dfu if not sector_sel else dfu[dfu['sector'].isin(sector_sel)]
        picks = st.multiselect("Semboller (10–30 önerilir)", pool['symbol'].astype(str).tolist(), max_selections=30, key="port_picks")
        method = st.selectbox("Yöntem", ["equal","meanvar","riskparity"], index=0, key="port_method")
        reb = st.selectbox("Rebalans", ["daily","weekly","monthly"], index=1, key="port_reb")
        fee = st.number_input("Ücret (bp)", 0, 100, 10, key="port_fee")
        slp = st.number_input("Slippage (bp)", 0, 100, 5, key="port_slp")
        if st.button("Backtest Çalıştır", key="port_run"):
            rets = []
            for s in picks[:30]:
                dfp = fetch_history(s, start=cfg['data']['start_date'])
                if dfp is None or dfp.empty:
                    continue
                rets.append(dfp['Close'].pct_change().rename(s))
            if not rets:
                st.error("Getiri serileri alınamadı.")
            else:
                R = pd.concat(rets, axis=1).dropna()
                eq, w = backtest_multi(R, method=method, rebalance=reb, fee_bp=fee, slippage_bp=slp)
                st.line_chart(eq.rename("Portföy"))
                st.write("Son ağırlıklar:", w)

# -------- Kripto (Binance) --------
with tabs[5]:
    st.subheader("Binance — Portföy optimizasyonu + backtest")
    quote    = st.selectbox("Quote", ["USDT","USDC","BUSD"], index=0, key="bin_quote")
    interval = st.selectbox("Periyot", ["1d","4h","1h"], index=0, key="bin_interval")
    bars     = st.slider("Bar sayısı", 200, 1000, 365, 50, key="bin_bars")
    topn     = st.slider("Sembol sayısı (ilk N)", 5, 200, 30, 5, key="bin_topn")
    method   = st.selectbox("Yöntem", ["equal","meanvar","riskparity"], index=0, key="bin_method")
    reb      = st.selectbox("Rebalans", ["daily","weekly","monthly"], index=0, key="bin_reb")
    fee      = st.number_input("Ücret (bp)", 0, 100, 8, key="bin_fee")
    slp      = st.number_input("Slippage (bp)", 0, 100, 5, key="bin_slp")
    if st.button("Veriyi çek ve backtest et", key="bin_run"):
        with st.spinner("Semboller ve fiyatlar alınıyor..."):
            R, syms = build_returns_matrix(quote=quote, interval=interval, limit=bars, top_n=topn)
        if R is None or R.empty:
            st.error("Veri alınamadı.")
        else:
            eq, w = optimize_and_backtest(R, method=method, rebalance=reb, fee_bp=fee, slippage_bp=slp)
            st.line_chart(eq.rename("Kripto Portföy"))
            st.write("Ağırlıklar:", w.rename("weight"))
