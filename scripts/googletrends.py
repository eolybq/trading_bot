from pytrends.request import TrendReq

# Inicializace API klienta
pytrends = TrendReq(hl='en-US', tz=360)

# Vytvoření požadavku (např. pro 'Bitcoin')
keywords = ["Bitcoin", "Ethereum", "Litecoin", "crypto", "cryptocurrency"]
pytrends.build_payload(keywords, timeframe="2023-01-01 2023-12-31", geo="")

# Získání dat o zájmu v čase
interest_over_time = pytrends.interest_over_time()
print("Interest over time:")
print(interest_over_time)
