from src.ingestion.load_data import DataIngestion

loader = DataIngestion("data/raw/titanic.csv")

df = loader.ingest()

print(df.head())