# 📊 Sales Data API

> API Flask per la generazione di dati di vendita simulati, deployata su Render.

## 🔗 Link API
**URL Base**: `https://sales-data-api.onrender.com`  
**Endpoint Disponibile**: `/generate-sales`

## 📋 Descrizione

Questa API genera dataset simulati di vendite, inclusi:
- Prodotti venduti
- Regione geografica
- Data di vendita
- Valore delle vendite
- Profitto netto

## 🚀 Endpoint

### Generate Random Sales Data
```http
GET /generate-sales?num_records={number}
```

**URL Completo di Esempio:**
```
https://sales-data-api.onrender.com/generate-sales?num_records=1000
```

**Parametri Query:**
- `num_records` (opzionale): Numero di record da generare (default: 1000, max: 100,000)

**Response Format**: JSON

---

## 📊 Struttura dei Dati

Ogni record contiene i seguenti campi:
```json
{
    "ID": 1,
    "Date": "2024-11-27",
    "Product": "Laptop",
    "Region": "North",
    "Sales": 543.21,
    "Profit": 123.45
}
```

### Dettagli dei campi:
- **ID**: Identificativo univoco del record
- **Date**: Data casuale dell'ultimo anno
- **Product**: Prodotto venduto (es. Laptop, Smartphone, ecc.)
- **Region**: Regione di vendita (North, South, East, West)
- **Sales**: Valore totale delle vendite (in USD, 100-1000)
- **Profit**: Valore netto del profitto (in USD, 10-200)

---

## 💻 Esempi di Utilizzo

### Python con Requests
```python
import requests
import pandas as pd

# Genera 1000 record di dati di vendita
response = requests.get("https://sales-data-api.onrender.com/generate-sales?num_records=1000")
data = response.json()

# Converti in DataFrame
df = pd.DataFrame(data)
print(df.head())
```

### JavaScript/Fetch
```javascript
fetch('https://sales-data-api.onrender.com/generate-sales?num_records=1000')
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

---

## ⚠️ Limiti
- Massimo 100,000 record per richiesta
- API gratuita con limiti di rate su Render
- Dati generati casualmente per scopi dimostrativi
- L'istanza gratuita su Render può andare in sleep dopo periodi di inattività, causando un ritardo nella prima richiesta

---

## 🔒 Note sulla Sicurezza
- I dati sono simulati e non contengono informazioni personali reali
- Non utilizzare per scopi di analisi o vendita reali
- API pubblica senza autenticazione

---

## 📝 Note di Sviluppo
- Implementata con Flask
- Deployata su Render
- Genera dati casuali ma realistici
- Supporta CORS per utilizzo frontend

---

## 🤝 Contributi
Per suggerimenti o problemi, apri una issue su GitHub o contatta il team di sviluppo.

---

Sviluppato con ❤️ per scopi didattici e dimostrativi.

---
