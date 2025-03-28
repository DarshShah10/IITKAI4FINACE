import yfinance as yf
import json
import os

def get_financial_statements(ticker, year):
    stock = yf.Ticker(ticker)
    
    # Fetch financial statements
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    
    financial_data = {
        "INCOME_STATEMENT": {},
        "BALANCE_SHEET": {},
        "CASH_FLOW": {}
    }
    
    # Convert columns to strings for comparison
    income_statement.columns = income_statement.columns.strftime('%Y')
    balance_sheet.columns = balance_sheet.columns.strftime('%Y')
    cash_flow.columns = cash_flow.columns.strftime('%Y')
    
    if str(year) in income_statement.columns:
        financial_data["INCOME_STATEMENT"] = income_statement[str(year)].to_dict()
    else:
        financial_data["INCOME_STATEMENT"] = {"error": f"No data for {year}"}
    
    if str(year) in balance_sheet.columns:
        financial_data["BALANCE_SHEET"] = balance_sheet[str(year)].to_dict()
    else:
        financial_data["BALANCE_SHEET"] = {"error": f"No data for {year}"}
    
    if str(year) in cash_flow.columns:
        financial_data["CASH_FLOW"] = cash_flow[str(year)].to_dict()
    else:
        financial_data["CASH_FLOW"] = {"error": f"No data for {year}"}
    
    return financial_data

def save_to_json(data, filename):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Financial data saved to {filename}")

def main():
    ticker = "HDFCBANK.NS"
    year = 2024
    
    financials = get_financial_statements(ticker, year)
    
    # Save to JSON file
    filename = f"{ticker}_{year}_financials.json"
    save_to_json(financials, filename)
    
    # Print formatted JSON output
    print(json.dumps(financials, indent=4))

if __name__ == "__main__":
    main()