import os, yaml
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_fetcher import fetch_history
from model_trainer import load_ensemble, make_features, predict_proba_ensemble
from signal_engine import news_sentiment_score, combine_signal
from universe_fetcher import fetch_bist_all, fetch_sp500_wiki, fetch_nasdaq_all, get_cached_universes
from portfolio_backtest import backtest_multi
from binance_portfolio import build_returns_matrix, optimize_and_backtest

load_dotenv()
st.set_page_config(layout="wide", page_title="Mini-Terminal Pro — ULTIMATE V9")

# Inject premium CSS
with open("assets/style.css","r",encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 style="margin-bottom:0.2rem;">Mini-Terminal Pro — ULTIMATE V9</h1>', unsafe_allow_html=True)
st.caption("Premium koyu tema • Neon vurgu • Terminal görünümü")

with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

tabs = st.tabs([
    "Evren Yönetimi",
    "Keşfet & Grafik",
    "Yapay Zeka Tahmin",
    "News Trading (Canlı)",
    "Portföy (Hisse)",
    "Kripto (Binance)"
])

# -------- Evren Yönetimi --------
with tabs[0]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("BIST + S&P500 + NASDAQ — Evren Güncelle")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("BIST: Hepsini Çek (Finnhub)"):
            try:
                df = fetch_bist_all(write_csv=True)
                st.success(f"BIST güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    with c2:
        if st.button("S&P 500: Wikipedia'dan Güncelle"):
            try:
                df = fetch_sp500_wiki(write_csv=True)
                st.success(f"S&P 500 güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    with c3:
        if st.button("NASDAQ: Listeyi Güncelle"):
            try:
                df = fetch_nasdaq_all(write_csv=True)
                st.success(f"NASDAQ güncellendi: {len(df)} kayıt")
            except Exception as e:
                st.error(f"Hata: {e}")
    uni = get_cached_universes()
    st.write({k: len(v) for k,v in uni.items()})
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Keşfet & Grafik --------
with tabs[1]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("Sektör filtreli grafik")
    uni = get_cached_universes()
    src = st.selectbox("Kaynak", ["bist","sp500","nasdaq"])
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        sector_list = sorted([s for s in dfu['sector'].dropna().unique().tolist() if s])
        sector_sel = st.multiselect("Sektör filtresi (opsiyonel)", sector_list)
        dff = dfu if not sector_sel else dfu[dfu['sector'].isin(sector_sel)]
        symbol = st.selectbox("Sembol", dff['symbol'].tolist())
        if symbol:
            dfp = fetch_history(symbol, start=cfg['data']['start_date'])
            if dfp is None or dfp.empty:
                st.warning("Veri bulunamadı.")
            else:
                fig = go.Figure(data=[go.Candlestick(x=dfp.index, open=dfp['Open'], high=dfp['High'], low=dfp['Low'], close=dfp['Close'])])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dfp.tail(10))
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Yapay Zeka Tahmin --------
with tabs[2]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("YUKARI / AŞAĞI tahmini ve güven skoru")
    uni = get_cached_universes()
    src = st.selectbox("Kaynak (tahmin)", ["bist","sp500","nasdaq"], key="predsrc")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        symbol = st.selectbox("Sembol (tahmin)", dfu['symbol'].tolist(), key="predsym")
        models = load_ensemble()
        if not models:
            st.info("Model yok. `train_all.py` ile hızlı eğitim yapabilir veya `models/` klasörüne .pkl koyabilirsiniz.")
        else:
            df = fetch_history(symbol, start=cfg['data']['start_date'])
            if df is None or df.empty:
                st.warning("Veri yok.")
            else:
                feat = make_features(df)
                prob = predict_proba_ensemble(models, feat[cfg['model']['features']].iloc[-1:]) or 0.5
                senti, _ = news_sentiment_score(symbol, hours=cfg['signal']['sentiment_lookback_days']*8)
                score = combine_signal(prob, senti, cfg)
                label = "YUKARI" if prob >= 0.5 else "AŞAĞI"
                st.metric("Tahmin", label, delta=f"Güven: {prob:.2f}")
                st.metric("Haber Sentiment", f"{senti:.3f}")
                st.metric("Kombine Skor", f"{score:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------- News Trading (Canlı) --------
with tabs[3]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("Anlık Haber Akışı — Sentiment Momentum")
    st.markdown(f"<meta http-equiv='refresh' content='{cfg['signal']['news_refresh_seconds']}'>", unsafe_allow_html=True)
    uni = get_cached_universes()
    src = st.selectbox("Kaynak (haber)", ["bist","sp500","nasdaq"], key="newssrc")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        symbol = st.selectbox("Sembol (haber)", dfu['symbol'].tolist(), key="newssym")
        hours = st.slider("Son kaç saat", 1, 24, 6)
        senti_now, news_df = news_sentiment_score(symbol, hours=hours)
        prev_senti, _ = news_sentiment_score(symbol, hours=min(hours*2,24))
        trend = "↑ Artış" if senti_now > prev_senti else ("↓ Düşüş" if senti_now < prev_senti else "→ Durağan")
        st.metric("Anlık Haber Sentiment", f"{senti_now:.3f}", delta=trend)
        st.caption(f"Otomatik yenileme: {cfg['signal']['news_refresh_seconds']} sn")
        if news_df is None or news_df.empty:
            st.info("Bu pencerede haber bulunamadı.")
        else:
            st.dataframe(news_df[['datetime','headline','score']])
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Portföy (Hisse) --------
with tabs[4]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("Sektör bazlı portföy optimizasyonu + maliyetli backtest")
    uni = get_cached_universes()
    src = st.selectbox("Kaynak (portföy)", ["bist","sp500","nasdaq"], key="portsrc")
    dfu = uni.get(src, pd.DataFrame())
    if dfu is None or dfu.empty:
        st.info("Önce evreni güncelle.")
    else:
        sector_list = sorted([s for s in dfu['sector'].dropna().unique().tolist() if s])
        sector_sel = st.multiselect("Sektör filtre (opsiyonel)", sector_list)
        pool = dfu if not sector_sel else dfu[dfu['sector'].isin(sector_sel)]
        picks = st.multiselect("Semboller (10-30 önerilir)", pool['symbol'].tolist(), max_selections=30)
        method = st.selectbox("Yöntem", ["equal","meanvar","riskparity"])
        reb = st.selectbox("Rebalans", ["daily","weekly","monthly"], index=1)
        fee = st.number_input("Ücret (bp)", 0, 100, 10)
        slp = st.number_input("Slippage (bp)", 0, 100, 5)
        if st.button("Backtest Çalıştır"):
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
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Kripto (Binance) --------
with tabs[5]:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("Binance — Portföy optimizasyonu + backtest")
    from binance_portfolio import build_returns_matrix, optimize_and_backtest
    quote = st.selectbox("Quote", ["USDT","USDC","BUSD"], index=0)
    interval = st.selectbox("Periyot", ["1d","4h","1h"], index=0)
    bars = st.slider("Bar sayısı", 200, 1000, 365, 50)
    topn = st.slider("Sembol sayısı (ilk N)", 5, 200, 30, 5)
    method = st.selectbox("Yöntem", ["equal","meanvar","riskparity"], index=0)
    reb = st.selectbox("Rebalans", ["daily","weekly","monthly"], index=0)
    fee = st.number_input("Ücret (bp)", 0, 100, 8)
    slp = st.number_input("Slippage (bp)", 0, 100, 5)
    if st.button("Veriyi çek ve backtest et"):
        with st.spinner("Semboller ve fiyatlar alınıyor..."):
            R, syms = build_returns_matrix(quote=quote, interval=interval, limit=bars, top_n=topn)
        if R is None or R.empty:
            st.error("Veri alınamadı.")
        else:
            eq, w = optimize_and_backtest(R, method=method, rebalance=reb, fee_bp=fee, slippage_bp=slp)
            st.line_chart(eq.rename("Kripto Portföy"))
            st.write("Ağırlıklar:", w.rename("weight"))
    st.markdown('</div>', unsafe_allow_html=True)
