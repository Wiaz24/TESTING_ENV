import yfinance as yf
import os
import pandas as pd
from tqdm import tqdm

directory = 'tickers'  # Podaj ścieżkę do katalogu
adj_prices = False
os.makedirs(directory, exist_ok=True)

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
    'JNK',   # SPDR Bloomberg High Yield Bond ETF - od 2007
    
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

for ticker in tqdm(etfs):

    ohclv_data = yf.download(ticker, interval='1d', auto_adjust=adj_prices, progress=False)
    
    if ohclv_data.empty or len(ohclv_data.index) < 10:
        print(f'Brak danych dla {ticker}')
        continue
    # Save in the correct format
    ohclv_data.to_csv(f'{directory}/{ticker}.csv', index=True, index_label='Date')

# Save correct headers
def replace_first_three_lines(file_path, new_line):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Zamiana pierwszych trzech linii na nową linię
    updated_lines = [new_line + '\n'] + lines[3:]
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(updated_lines)

def process_csv_files(directory, new_line):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            replace_first_three_lines(file_path, new_line)
            print(f'Przetworzono plik: {filename}')

new_line = 'Date,Adj Close,Close,High,Low,Open,Volume'  # Podaj nową linię
process_csv_files(directory, new_line)
