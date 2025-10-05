# Implementação de Black-Scholes

# Importando bibliotecas necessárias
import numpy as np
from scipy.stats import norm 

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


