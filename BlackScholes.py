# Implementação de Black-Scholes

# Importando bibliotecas necessárias
import numpy as np
from scipy.stats import norm 
from scipy.optimize import brentq # <-- ADICIONE ESTA
import warnings # <-- ADICIONE ESTA

# Definindo a função de Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calcula o preço de uma opção usando o modelo Black-Scholes.
    
    Parâmetros:
    S : float : Preço atual do ativo subjacente
    K : float : Preço de exercício da opção
    T : float : Tempo até o vencimento (em anos)
    r : float : Taxa de juros livre de risco (anualizada)
    sigma : float : Volatilidade do ativo subjacente (anualizada)
    option_type : str : Tipo da opção ('call' ou 'put')
    
    Retorna:
    float : Preço da opção
    """

    # Cálculo dos parâmetros d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("option_type deve ser 'call' ou 'put'")
    
    return price

# Exemplo de uso:

S = 100      # Preço atual do ativo
K = 115      # Preço de exercício
T = 1        # Tempo até o vencimento (1 ano)
r = 0.15     # Taxa de juros livre de risco (15%)
sigma = 0.2  # Volatilidade (20%)

call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

print(f"Preço da Call: {call_price:.2f}")
print(f"Preço da Put: {put_price:.2f}")

# --- CÁLCULO DAS GREGAS ---

def delta(S, K, T, r, sigma, option_type='call'):
    """ Calcula o Delta de uma opção. """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type deve ser 'call' ou 'put'")

def gamma(S, K, T, r, sigma):
    """ Calcula o Gamma de uma opção. """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """ Calcula o Vega de uma opção. """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # Retorna o Vega por 1 p.p. de mudança na vol (multiplicado por 0.01)
    return (S * norm.pdf(d1) * np.sqrt(T)) * 0.01

def theta(S, K, T, r, sigma, option_type='call'):
    """ Calcula o Theta de uma opção (por dia). """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        theta_calc = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        theta_calc = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type deve ser 'call' ou 'put'")
    
    # Retorna o Theta diário (dividido por 365)
    return theta_calc / 365.0

def rho(S, K, T, r, sigma, option_type='call'):
    """ Calcula o Rho de uma opção. """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Retorna o Rho por 1 p.p. de mudança nos juros (multiplicado por 0.01)
    if option_type == 'call':
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) * 0.01
    elif option_type == 'put':
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) * 0.01
    else:
        raise ValueError("option_type deve ser 'call' ou 'put'")
    
    # --- CÁLCULO DE VOLATILIDADE IMPLÍCITA ---

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calcula a Volatilidade Implícita (VI) usando o método brentq.
    """
    
    # Função objetivo: a diferença entre o preço do modelo e o preço de mercado
    def objective_function(sigma):
        # Evita valores de sigma não realistas
        if sigma <= 0:
            return -market_price
        
        price = black_scholes(S, K, T, r, sigma, option_type)
        
        # Se o preço for NaN (devido a sigma muito baixo), retorna erro
        if np.isnan(price):
            return 1e6 # Um número grande para penalizar
            
        return price - market_price

    # Tenta encontrar a VI em um intervalo razoável (0.1% a 500%)
    # brentq é um solver rápido que encontra a raiz (onde a função == 0)
    try:
        # Ignora avisos comuns do solver que poluem o terminal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            vi = brentq(objective_function, 1e-6, 5.0) # Busca entre 0.0001% e 500%
        
        # Se o solver retornar um valor muito próximo de zero, é provável lixo
        if vi < 1e-5:
            return np.nan
            
        return vi
    
    except ValueError:
        # Se não encontrar a raiz (ex: preço de mercado impossível/arbitragem)
        return np.nan