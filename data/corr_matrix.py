import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skfolio.distance import PearsonDistance
from skfolio.preprocessing import prices_to_returns

# Lista ETF-ów
etfs = [
    # Duża kapitalizacja
    'SPY',   # S&P 500 - od 1993
    'VV',    # Vanguard Large-Cap - od 2004
    'QQQ',   # Invesco QQQ (Nasdaq) - od 1999
    'EZU',   # iShares MSCI Eurozone ETF (zamiast VGK) - od 2000
    'EWJ',   # iShares MSCI Japan - od 1996
    
    # Średnia kapitalizacja
    'VO',    # Vanguard Mid-Cap - od 2004
    'IJH',   # iShares Core S&P Mid-Cap - od 2000
    'MDY',   # SPDR S&P MidCap 400 (zamiast IVOO) - od 1995
    
    # Mała kapitalizacja
    'VB',    # Vanguard Small-Cap - od 2004
    'IJR',   # iShares Core S&P Small-Cap - od 2000
    'VSS',   # Vanguard FTSE All-World ex-US Small-Cap - od 2009
    
    # Rynki rozwinięte
    'VEA',   # Vanguard FTSE Developed Markets - od 2007
    'EFA',   # iShares MSCI EAFE - od 2001
    'IEV',   # iShares Europe ETF (zamiast HEDJ) - od 2000
    
    # Rynki wschodzące
    'VWO',   # Vanguard FTSE Emerging Markets - od 2005
    'EEM',   # iShares MSCI Emerging Markets - od 2003
    'EPI',   # WisdomTree India Earnings Fund (zamiast INDA) - od 2008
    'FXI',   # iShares China Large-Cap ETF (zamiast MCHI) - od 2004
    'EWZ',   # iShares MSCI Brazil - od 2000
    
    # Obligacje skarbowe
    'AGG',   # iShares Core U.S. Aggregate Bond - od 2003
    'BND',   # Vanguard Total Bond Market - od 2007
    'IEF',   # iShares 7-10 Year Treasury Bond - od 2002
    'TLT',   # iShares 20+ Year Treasury Bond - od 2002
    'BWX',   # SPDR Bloomberg International Treasury Bond (zamiast BNDX) - od 2007
    
    # Obligacje korporacyjne
    'LQD',   # iShares iBoxx $ Investment Grade Corporate Bond - od 2002
    'HYG',   # iShares iBoxx $ High Yield Corporate Bond - od 2007
    'VCSH',  # Vanguard Short-Term Corporate Bond - od 2009
    
    # Sektorowe
    'XLF',   # Financial Select Sector SPDR - od 1998
    'XLK',   # Technology Select Sector SPDR - od 1998
    'XLE',   # Energy Select Sector SPDR - od 1998
    'XLV',   # Health Care Select Sector SPDR - od 1998
    'XLP',   # Consumer Staples Select Sector SPDR - od 1998
    'XLY',   # Consumer Discretionary Select Sector SPDR - od 1998
    'XLU',   # Utilities Select Sector SPDR - od 1998
    'XLB',   # Materials Select Sector SPDR - od 1998
    
    # Alternatywne klasy aktywów
    'GLD',   # SPDR Gold Shares - od 2004
    'SLV',   # iShares Silver Trust - od 2006
    'VNQ',   # Vanguard Real Estate - od 2004
    'RWX',   # SPDR Dow Jones International Real Estate (zamiast VNQI) - od 2006
    'GSG',   # iShares S&P GSCI Commodity-Indexed Trust - od 2006
    'USO',   # United States Oil Fund - od 2006
    'UNG',   # United States Natural Gas Fund - od 2007
    
    # Strategie czynnikowe
    'IWF',   # iShares Russell 1000 Growth ETF (zamiast MTUM) - od 2000
    'IWD',   # iShares Russell 1000 Value ETF (zamiast VLUE) - od 2000
    'PWV',   # Invesco Dynamic Large Cap Value ETF (zamiast QUAL) - od 2005
    'IWM',   # iShares Russell 2000 ETF (zamiast SIZE) - od 2000
    'POWA',  # Invesco Bloomberg Pricing Power - od 2007
    
    # Dodatkowe ETF-y do uzupełnienia listy
    'DIA',   # SPDR Dow Jones Industrial Average ETF - od 1998
    'IYR',   # iShares U.S. Real Estate ETF - od 2000
    'KRE'    # SPDR S&P Regional Banking ETF - od 2006
]

# Katalog z plikami CSV
csv_directory = 'tickers'  # zmień na swój katalog jeśli pliki są w innym miejscu

# Wczytanie danych i połączenie ich w jeden DataFrame
prices_data = {}

for ticker in etfs:
    file_path = os.path.join(csv_directory, f"{ticker}.csv")
    
    if os.path.exists(file_path):
        try:
            # Wczytaj dane
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Dodaj kolumnę z ceną zamknięcia do głównego DataFrame
            prices_data[ticker] = df['Close']
            
            print(f"Wczytano dane dla {ticker}")
        except Exception as e:
            print(f"Błąd przy wczytywaniu {ticker}: {str(e)}")
    else:
        print(f"Nie znaleziono pliku dla {ticker}")

# Utwórz DataFrame z cenami zamknięcia
prices_df = pd.DataFrame(prices_data)

# Znajdź wspólny okres dla wszystkich ETF-ów
prices_df = prices_df.dropna()

print(f"\nLiczba dni z danymi dla wszystkich ETF-ów: {len(prices_df)}")
print(f"Okres danych: od {prices_df.index.min().date()} do {prices_df.index.max().date()}")

# Konwersja cen na zwroty
returns_df = prices_to_returns(prices_df)

# Użyj skfolio do obliczenia macierzy korelacji
distance_model = PearsonDistance()
distance_model.fit(returns_df)

# Pobierz macierz korelacji
correlation_matrix = distance_model.codependence_

# Wizualizacja macierzy korelacji
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Rysuj macierz korelacji używając seaborn dla lepszego efektu wizualnego
sns.heatmap(correlation_matrix, 
            mask=mask,
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            xticklabels=etfs,
            yticklabels=etfs,
            annot=False)

plt.title('Macierz korelacji zwrotów ETF-ów', fontsize=16)
plt.tight_layout()
plt.savefig('etf_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Zapisano macierz korelacji jako 'etf_correlation_matrix.png'")

# Zapisz macierz korelacji do CSV
correlation_df = pd.DataFrame(correlation_matrix, columns=etfs, index=etfs)
correlation_df.to_csv('etf_correlation_matrix.csv')
print("Zapisano dane macierzy korelacji do 'etf_correlation_matrix.csv'")

# Analiza statystyczna korelacji
# Znajdź pary ETF-ów o najwyższej korelacji
high_corr = []
for i in range(len(etfs)):
    for j in range(i+1, len(etfs)):
        corr = correlation_matrix[i, j]
        high_corr.append((etfs[i], etfs[j], corr))

high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nNajwyższe korelacje:")
for i, (etf1, etf2, corr) in enumerate(high_corr[:15]):
    print(f"{i+1}. {etf1} - {etf2}: {corr:.4f}")

print("\nNajniższe korelacje:")
for i, (etf1, etf2, corr) in enumerate(high_corr[-15:]):
    print(f"{i+1}. {etf1} - {etf2}: {corr:.4f}")

# Oblicz średnią korelację dla każdego ETF
avg_corr = {}
for i, etf in enumerate(etfs):
    # Pomiń korelację ETF z samym sobą (która wynosi 1)
    corrs = [correlation_matrix[i, j] for j in range(len(etfs)) if i != j]
    avg_corr[etf] = sum(corrs) / len(corrs)

avg_corr = {k: v for k, v in sorted(avg_corr.items(), key=lambda item: item[1], reverse=True)}

print("\nŚrednie korelacje każdego ETF z pozostałymi:")
for i, (etf, corr) in enumerate(avg_corr.items()):
    print(f"{i+1}. {etf}: {corr:.4f}")

# Zapisz listę ETF-ów posortowaną wg średniej korelacji
sorted_etfs_df = pd.DataFrame(list(avg_corr.items()), columns=['ETF', 'Średnia korelacja'])
sorted_etfs_df.to_csv('etf_average_correlation.csv', index=False)
print("Zapisano średnie korelacje do 'etf_average_correlation.csv'")