DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'text2sql_reader') THEN
        CREATE ROLE text2sql_reader LOGIN;
    END IF;
END
$$;

ALTER ROLE text2sql_reader PASSWORD :'reader_password';

ALTER ROLE text2sql_reader SET default_transaction_read_only = on;
ALTER ROLE text2sql_reader SET statement_timeout = '5s';
GRANT CONNECT ON DATABASE text2sql_books TO text2sql_reader;
GRANT USAGE ON SCHEMA public TO text2sql_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO text2sql_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO text2sql_reader;
