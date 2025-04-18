import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# Lista tickerów ETF
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
csv_directory = 'tickers'  # zmień na swój katalog, jeśli pliki są w innym miejscu

# Słownik na zakresy dat
date_ranges = {}
missing_files = []

# Sprawdź dostępność danych dla każdego ETF
for ticker in etfs:
    file_path = os.path.join(csv_directory, f"{ticker}.csv")
    
    if os.path.exists(file_path):
        try:
            # Wczytaj dane
            df = pd.read_csv(file_path)
            
            # Konwersja dat
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Znajdź pierwszą i ostatnią datę
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            
            # Zapisz zakres dat
            date_ranges[ticker] = (start_date, end_date)
            
            print(f"Przetworzono {ticker}: od {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {ticker}: {str(e)}")
            missing_files.append(ticker)
    else:
        print(f"Nie znaleziono pliku dla {ticker}")
        missing_files.append(ticker)

# Posortuj ETF-y według daty początkowej
sorted_etfs = sorted(date_ranges.keys(), key=lambda x: date_ranges[x][0])

# Przygotuj dane do wykresu
y_positions = range(len(sorted_etfs))
etf_labels = sorted_etfs
start_dates = [date_ranges[ticker][0] for ticker in sorted_etfs]
end_dates = [date_ranges[ticker][1] for ticker in sorted_etfs]
durations = [(end_dates[i] - start_dates[i]).days for i in range(len(sorted_etfs))]

# Utwórz figure z odpowiednim rozmiarem
plt.figure(figsize=(15, 12))

# Narysuj poziome linie dla każdego ETF
for i, ticker in enumerate(sorted_etfs):
    start_date = date_ranges[ticker][0]
    end_date = date_ranges[ticker][1]
    plt.hlines(y=i, xmin=start_date, xmax=end_date, linewidth=6, color='steelblue')
    
    # Dodaj kropki na początku i końcu
    plt.plot(start_date, i, 'o', markersize=8, color='darkblue')
    plt.plot(end_date, i, 'o', markersize=8, color='darkblue')

# Ustaw etykiety osi Y (tickery)
plt.yticks(y_positions, etf_labels)

# Formatowanie osi X (daty)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)

# Dodaj siatkę dla lepszej czytelności
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Dodaj tytuł i etykiety osi
plt.title('Zakres dostępności danych dla ETF-ów', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Ticker ETF', fontsize=12)

# Dodaj informacje o brakujących plikach
if missing_files:
    missing_text = f"Brakujące pliki ({len(missing_files)}): {', '.join(missing_files)}"
    plt.figtext(0.5, 0.01, missing_text, ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Dostosuj układ
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Pozostaw miejsce na informacje o brakujących plikach

# Zapisz wykres do pliku
plt.savefig('etf_data_availability.png', dpi=300, bbox_inches='tight')
print("Wykres został zapisany jako 'etf_data_availability.png'")

# Podstawowe statystyki
min_start_date = min(start_dates)
max_end_date = max(end_dates)
common_start_date = max(start_dates)
common_end_date = min(end_dates)

print("\nStatystyki zakresu danych:")
print(f"Najwcześniejsza data rozpoczęcia: {min_start_date.strftime('%Y-%m-%d')}")
print(f"Najpóźniejsza data zakończenia: {max_end_date.strftime('%Y-%m-%d')}")
print(f"Wspólny okres dla wszystkich ETF-ów: {common_start_date.strftime('%Y-%m-%d')} - {common_end_date.strftime('%Y-%m-%d')}")
print(f"Długość wspólnego okresu: {(common_end_date - common_start_date).days} dni")

# Zapisz wyniki do pliku CSV
results = []
for ticker in sorted_etfs:
    start_date = date_ranges[ticker][0]
    end_date = date_ranges[ticker][1]
    duration = (end_date - start_date).days
    results.append({
        'Ticker': ticker,
        'Start Date': start_date.strftime('%Y-%m-%d'),
        'End Date': end_date.strftime('%Y-%m-%d'),
        'Duration (days)': duration
    })

# Zapisz do CSV
results_df = pd.DataFrame(results)
results_df.to_csv('etf_data_summary.csv', index=False)
print("Podsumowanie zostało zapisane do pliku 'etf_data_summary.csv'")