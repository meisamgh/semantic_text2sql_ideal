CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS authors (
    author_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    country TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publishers (
    publisher_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    country TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    category_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS books (
    book_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    publication_date DATE NOT NULL,
    retail_price NUMERIC(10, 2) NOT NULL CHECK (retail_price >= 0),
    author_id BIGINT NOT NULL REFERENCES authors(author_id),
    publisher_id BIGINT NOT NULL REFERENCES publishers(publisher_id),
    category_id BIGINT NOT NULL REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    book_id BIGINT NOT NULL REFERENCES books(book_id),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS books_publication_date_idx ON books(publication_date);
CREATE INDEX IF NOT EXISTS books_retail_price_idx ON books(retail_price);
CREATE INDEX IF NOT EXISTS reviews_book_id_idx ON reviews(book_id);
