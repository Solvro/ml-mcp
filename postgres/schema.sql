CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE offers (
  id SERIAL PRIMARY KEY,
  link TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  company TEXT,
  location TEXT,
  contract_type TEXT,
  date_posted DATE,
  date_closing DATE,
  source TEXT,
  description TEXT,
  embedding vector(1536)
);