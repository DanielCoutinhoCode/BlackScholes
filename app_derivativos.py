# app_derivativos.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import os

# --- IMPORTANDO O SEU MOTOR DE CÁLCULO ---
# Importa TODAS as funções do seu outro arquivo
try:
    from BlackScholes import (
        black_scholes, 
        delta, 
        gamma, 
        vega, 
        theta, 
        rho, 
        implied_volatility
    )
except ImportError:
    st.error("Erro: O arquivo 'BlackScholes.py' não foi encontrado. Certifique-se de que ele está na mesma pasta que o 'app_derivativos.py'.")
    st.stop()
except Exception as e:
    st.error(f"Um erro inesperado ocorreu ao importar 'BlackScholes.py': {e}")
    st.stop()


# --- 1. CONFIGURAÇÃO DA PÁGINA E SIDEBAR ---
st.set_page_config(
    layout="wide", 
    page_title="Análise de Derivativos",
    page_icon="📈"
)

st.title("Risk Dashboard - Precificação e Análise de Gregas")
st.markdown("Uma ferramenta interativa para calcular Volatilidade Implícita e analisar as Gregas de opções.")

# --- SIDEBAR DE INPUTS ---
st.sidebar.header("Parâmetros da Opção")

option_type = st.sidebar.selectbox(
    "Tipo de Opção", 
    ('call', 'put'),
    help="Escolha 'call' (compra) ou 'put' (venda)."
)

S = st.sidebar.number_input(
    "Preço do Ativo (S)", 
    value=100.0, 
    min_value=1.0, 
    step=0.5,
    help="Preço atual do ativo subjacente (ex: PETR4)."
)

K = st.sidebar.number_input(
    "Preço de Exercício (K)", 
    value=100.0, 
    min_value=1.0,
    step=0.5,
    help="Preço pelo qual a opção pode ser exercida (Strike)."
)

T_days = st.sidebar.number_input(
    "Dias até o Vencimento (T)", 
    value=30, 
    min_value=1,
    help="Número de dias corridos até o vencimento da opção."
)
T = T_days / 365.0  # Converte dias para anos para o cálculo

r = st.sidebar.slider(
    "Taxa Livre de Risco (r) %", 
    0.0, 20.0, 10.0, 
    step=0.25,
    help="Taxa de juros anualizada (ex: Taxa Selic)."
)
r_calc = r / 100.0 # Converte % para decimal

st.sidebar.header("Cálculo de Volatilidade Implícita")
market_price = st.sidebar.number_input(
    "Preço de Mercado da Opção", 
    value=5.0, 
    min_value=0.01,
    step=0.01,
    help="O preço atual da opção negociada na bolsa (ex: PETR4 K100)."
)
st.sidebar.markdown("---")
st.sidebar.info(f"Executando com Python: {sys.version.split(' ')[0]}")

# --- 2. CÁLCULOS CENTRAIS ---
vi = implied_volatility(market_price, S, K, T, r_calc, option_type)

if np.isnan(vi):
    st.error(f"Não foi possível calcular a Volatilidade Implícita para o preço R$ {market_price}. O preço pode estar fora do limite de arbitragem (valor intrínseco).")
    st.stop()
else:
    # Calcula tudo com base na VI encontrada
    calc_price = black_scholes(S, K, T, r_calc, vi, option_type)
    calc_delta = delta(S, K, T, r_calc, vi, option_type)
    calc_gamma = gamma(S, K, T, r_calc, vi)
    calc_vega = vega(S, K, T, r_calc, vi)
    calc_theta = theta(S, K, T, r_calc, vi, option_type)
    calc_rho = rho(S, K, T, r_calc, vi, option_type)

# --- 3. LAYOUT COM ABAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Calculadora Principal", 
    "📈 Análise de Sensibilidade (2D)", 
    "🧊 Visualização 3D",
    " diagrama de Payoff"
])

# --- ABA 1: CALCULADORA PRINCIPAL ---
with tab1:
    st.header(f"Resultados para {option_type.upper()} - {T_days} dias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Precificação")
        st.metric(label="Volatilidade Implícita (VI)", value=f"{vi:.2%}")
        st.metric(label="Preço Calculado (B-S)", value=f"R$ {calc_price:.3f}", 
                  help=f"O preço de mercado inserido foi R$ {market_price:.2f}. A pequena diferença se deve ao arredondamento do solver.")

    with col2:
        st.subheader("Análise de Risco (Gregas)")
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Delta (Δ)", value=f"{calc_delta:.4f}", 
                  help=f"Exposição ao preço do ativo. Para cada R$ 1,00 que o ativo sobe, a opção se valoriza R$ {calc_delta:.2f}.")
        c2.metric(label="Gamma (Γ)", value=f"{calc_gamma:.4f}", 
                  help="A 'velocidade' do Delta. Mostra o quanto o Delta mudará para cada R$ 1,00 de movimento do ativo.")
        c3.metric(label="Vega", value=f"{calc_vega:.4f}", 
                  help=f"Sensibilidade à Volatilidade. A opção se valoriza R$ {calc_vega:.2f} para cada 1 p.p. de aumento na VI.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Theta (Θ) Diário", value=f"R$ {calc_theta:.4f}", 
                  help=f"Decaimento pelo tempo. A opção perde R$ {abs(calc_theta):.4f} por dia, mantendo o resto constante.")
        c2.metric(label="Rho (ρ)", value=f"{calc_rho:.4f}", 
                  help=f"Sensibilidade à taxa de juros. A opção se valoriza R$ {calc_rho:.2f} para cada 1 p.p. de aumento na taxa de juros.")
        c3.metric(label="Status", value="Calculado ✅")

# --- ABA 2: ANÁLISE DE SENSITIBILIDADE (2D) ---
with tab2:
    st.header("Sensibilidade das Gregas (Gráficos 2D)")
    st.markdown("Como as Gregas mudam conforme o preço do ativo (S) flutua, mantendo a VI e o tempo constantes.")
    
    # Gera um range de preços do ativo
    S_range = np.linspace(S * 0.7, S * 1.3, 100)
    
    # Calcula Gregas para o range
    delta_range = [delta(s_val, K, T, r_calc, vi, option_type) for s_val in S_range]
    gamma_range = [gamma(s_val, K, T, r_calc, vi) for s_val in S_range]
    theta_range = [theta(s_val, K, T, r_calc, vi, option_type) for s_val in S_range]
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_delta = go.Figure(data=go.Scatter(x=S_range, y=delta_range, mode='lines', line=dict(color='blue')))
        fig_delta.update_layout(title="Delta vs. Preço do Ativo (S)", 
                                xaxis_title="Preço do Ativo (S)", yaxis_title="Delta")
        fig_delta.add_vline(x=S, line_width=2, line_dash="dash", line_color="red", annotation_text="Preço Atual")
        st.plotly_chart(fig_delta, use_container_width=True)

    with col2:
        fig_gamma = go.Figure(data=go.Scatter(x=S_range, y=gamma_range, mode='lines', line=dict(color='green')))
        fig_gamma.update_layout(title="Gamma vs. Preço do Ativo (S)", 
                                xaxis_title="Preço do Ativo (S)", yaxis_title="Gamma")
        fig_gamma.add_vline(x=S, line_width=2, line_dash="dash", line_color="red", annotation_text="Preço Atual")
        st.plotly_chart(fig_gamma, use_container_width=True)
        
    col1, col2 = st.columns(2)
    with col1:
        fig_theta = go.Figure(data=go.Scatter(x=S_range, y=theta_range, mode='lines', line=dict(color='orange')))
        fig_theta.update_layout(title="Theta vs. Preço do Ativo (S)", 
                                xaxis_title="Preço do Ativo (S)", yaxis_title="Theta Diário")
        fig_theta.add_vline(x=S, line_width=2, line_dash="dash", line_color="red", annotation_text="Preço Atual")
        st.plotly_chart(fig_theta, use_container_width=True)
        
    with col2:
        # Gráfico de Decaimento do Tempo (Theta)
        T_range_days = np.arange(T_days, 0, -1)
        T_range_years = T_range_days / 365.0
        theta_decay = [theta(S, K, t_val, r_calc, vi, option_type) for t_val in T_range_years]

        fig_decay = go.Figure(data=go.Scatter(x=T_range_days, y=theta_decay, mode='lines', line=dict(color='red')))
        fig_decay.update_layout(title="Decaimento do Theta (Theta vs. Tempo)",
                                xaxis_title="Dias até o Vencimento", yaxis_title="Theta Diário")
        fig_decay.update_xaxes(autorange="reversed") # Começa de hoje até o vencimento
        st.plotly_chart(fig_decay, use_container_width=True)

# --- ABA 3: VISUALIZAÇÃO 3D ---
with tab3:
    st.header("Superfície de Risco (3D)")
    st.markdown("Como uma Grega (ex: Delta) muda com o **Preço do Ativo** e o **Tempo até Vencer**.")

    # Gera ranges de Tempo e Preço
    T_range_3D_years = np.linspace(T, 0.01, 20) # De hoje até o vencimento
    T_range_3D_days = (T_range_3D_years * 365).astype(int)
    S_range_3D = np.linspace(S * 0.7, S * 1.3, 20)
    T_grid, S_grid = np.meshgrid(T_range_3D_years, S_range_3D)
    
    # Calcula o Delta para cada ponto da grade
    delta_grid = np.zeros_like(T_grid)
    for i in range(len(S_range_3D)):
        for j in range(len(T_range_3D_years)):
            delta_grid[i, j] = delta(S_grid[i, j], K, T_grid[i, j], r_calc, vi, option_type)

    # Gráfico 3D com Matplotlib
    # (Plotly 3D é uma opção, mas Matplotlib é robusto para superfícies)
    fig_3d = plt.figure(figsize=(12, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid * 365, delta_grid, cmap=cm.viridis, rstride=1, cstride=1)
    
    ax.set_xlabel('Preço do Ativo (S)')
    ax.set_ylabel('Dias até Vencer')
    ax.set_zlabel('Delta')
    ax.set_title(f'Superfície do Delta - {option_type.upper()}')
    ax.view_init(30, -120) # Ajusta o ângulo de visão
    fig_3d.colorbar(surf, shrink=0.5, aspect=10, label="Valor do Delta")
    
    st.pyplot(fig_3d)
    st.caption("Esta visualização 3D mostra como o Delta (eixo Z) se 'achata' para 0 ou 1 (para calls) conforme o vencimento (eixo Y) se aproxima e o preço (eixo X) fica muito dentro ou fora do dinheiro.")

# --- ABA 4: DIAGRAMA DE PAYOFF ---
with tab4:
    st.header("Diagrama de Payoff no Vencimento")
    st.markdown("Este gráfico mostra seu perfil de Lucro & Perda (P&L) no dia do vencimento.")
    
    S_payoff = np.linspace(S * 0.5, S * 1.5, 100) # Range de preços no vencimento
    
    if option_type == 'call':
        # Lucro = Payoff no vencimento - Custo pago
        payoff = np.maximum(S_payoff - K, 0) - market_price 
    else: # put
        # Lucro = Payoff no vencimento - Custo pago
        payoff = np.maximum(K - S_payoff, 0) - market_price 

    fig_payoff = go.Figure(data=go.Scatter(x=S_payoff, y=payoff, mode='lines', line=dict(color='blue')))
    
    # Calcula o Ponto de Breakeven
    if option_type == 'call':
        breakeven = K + market_price
        max_loss = -market_price
    else:
        breakeven = K - market_price
        max_loss = -market_price

    fig_payoff.update_layout(
        title=f"Payoff de {option_type.upper()} (Custo de R$ {market_price:.2f})",
        xaxis_title="Preço do Ativo no Vencimento", 
        yaxis_title="Lucro / Prejuízo (P&L)"
    )
    
    # Linha de Breakeven (Lucro = 0)
    fig_payoff.add_hline(y=0, line_width=2, line_dash="dash", line_color="black")
    # Linha do Strike
    fig_payoff.add_vline(x=K, line_width=2, line_dash="dash", line_color="grey", 
                       annotation_text="Strike (K)", annotation_position="top left")
    # Linha do Breakeven
    fig_payoff.add_vline(x=breakeven, line_width=2, line_dash="dash", line_color="green", 
                       annotation_text="Breakeven", annotation_position="top right")
    # Linha de Perda Máxima
    fig_payoff.add_hline(y=max_loss, line_width=2, line_dash="dash", line_color="red", 
                       annotation_text="Perda Máxima", annotation_position="bottom left")
                       
    st.plotly_chart(fig_payoff, use_container_width=True)
    st.metric(label="Ponto de Breakeven no Vencimento", value=f"R$ {breakeven:.2f}")
    st.metric(label="Perda Máxima (Custo da Opção)", value=f"R$ {max_loss:.2f}")