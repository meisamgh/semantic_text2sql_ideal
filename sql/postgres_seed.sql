TRUNCATE reviews, books, categories, publishers, authors RESTART IDENTITY CASCADE;

INSERT INTO authors (first_name, last_name, country) VALUES
    ('George', 'Orwell', 'United Kingdom'),
    ('Ursula', 'Le Guin', 'United States'),
    ('Margaret', 'Atwood', 'Canada'),
    ('Terry', 'Pratchett', 'United Kingdom');

INSERT INTO publishers (name, country) VALUES
    ('Secker & Warburg', 'United Kingdom'),
    ('Ace Books', 'United States'),
    ('McClelland & Stewart', 'Canada'),
    ('Doubleday', 'United Kingdom');

INSERT INTO categories (name) VALUES
    ('Dystopian'), ('Science Fiction'), ('Fantasy');

INSERT INTO books
    (title, description, publication_date, retail_price, author_id, publisher_id, category_id)
VALUES
    ('Nineteen Eighty-Four', 'A society shaped by surveillance and totalitarian control.',
     '1949-06-08', 14.99, 1, 1, 1),
    ('The Dispossessed', 'An anarchist society contrasted with a wealthy neighbouring world.',
     '1974-05-01', 17.50, 2, 2, 2),
    ('The Handmaid''s Tale', 'A theocratic regime removes freedom and autonomy from women.',
     '1985-09-01', 16.25, 3, 3, 1),
    ('Guards! Guards!', 'A comic fantasy about civic duty, dragons and city politics.',
     '1989-11-09', 12.75, 4, 4, 3);

INSERT INTO reviews (book_id, rating, review_text) VALUES
    (1, 5, 'A powerful warning about surveillance and freedom.'),
    (1, 4, 'Dark, political and still relevant.'),
    (2, 5, 'Thoughtful social science fiction about equality.'),
    (3, 5, 'A disturbing exploration of autonomy and oppression.'),
    (4, 4, 'Funny fantasy with memorable characters.');
