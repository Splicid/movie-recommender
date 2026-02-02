import pandas as pd
import requests
import psycopg2
import io
import os

from psycopg2 import sql
from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate, train_test_split
from sklearn.datasets import fetch_openml, clear_data_home


def main():
    movie_data, rating_data = read_csv_data()

    df_data = pd.DataFrame(movie_data)
    df_ratings = pd.DataFrame(rating_data)
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

    merged_df = pd.merge(df_ratings, df_data, on="movieId", how="inner")
    try:
        with psycopg2.connect(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=os.getenv("DB_PORT", "5435"),
            connect_timeout=3
        ) as conn:
            init_db(conn)
            write_to_db(conn, merged_df, "user_movie_summary")
            recommendation_service(conn, 1, "user_movie_summary")
    except Exception as error:
        print(f"Main has the error: {error}")

def init_db(conn):
    table_name = "user_movie_summary"
    try:
        cursor = conn.cursor()

        query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            userId INTEGER, 
            movieId INTEGER, 
            rating NUMERIC(3,1), 
            timestamp TIMESTAMPTZ,
            title TEXT,
            genres TEXT
            );
        """).format(table=sql.Identifier(table_name))

        cursor.execute(query)
        conn.commit()
        print("Database initilized")

    except (Exception, psycopg2.DatabaseError) as error:
            print(f"Connection Failed: {error}")
    finally:
        cursor.close()

def write_to_db(conn, df, table_name):
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)

    try:
        cursor = conn.cursor()
        cursor.execute(sql.SQL("TRUNCATE TABLE {table}").format(
            table=sql.Identifier(table_name)
        ))
        cursor.copy_from(output, table_name, sep='\t', null='')
        conn.commit()
        print(f"Successfully wrote {len(df)} rows to {table_name}")
    except Exception as error:
        conn.rollback()
        print(f"Error writing to DB: {error}")
    finally:
        cursor.close()
    

def recommendation_service(conn, userId, table):
    try: 
        curr = conn.cursor()
        curr.execute(sql.SQL("SELECT userId, movieId, rating FROM {table}").format(
            table=sql.Identifier(table)))
        df = pd.DataFrame(curr.fetchall(), columns=['userId', 'movieId', 'rating'])

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df, reader)
        trainset = data.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)

        # Get a list of ALL unique movie IDs and titles from the DB
        curr.execute(sql.SQL("SELECT DISTINCT movieId, title FROM {table}").format(
            table=sql.Identifier(table)))
        all_movies = {row[0]: row[1] for row in curr.fetchall()}

        # Find movies this specific user has ALREADY rated
        curr.execute(sql.SQL("SELECT movieId FROM {table} WHERE userId = %s").format(
            table=sql.Identifier(table)), (userId,))
        rated_movies = set(row[0] for row in curr.fetchall())

        # Predict ratings for movies the user hasn't seen yet
        predictions = []
        for m_id, m_title in all_movies.items():
            if m_id not in rated_movies:
                est_rating = algo.predict(userId, m_id).est
                predictions.append((m_title, est_rating))

        # Sort by estimated rating and take the top 5
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_5 = predictions[:5]

        print(f"\n--- Top 5 Recommendations for User {userId} ---")
        for i, (title, score) in enumerate(top_5, 1):
            print(f"{i}. {title} (Predicted Rating: {score:.2f})")
            
        return top_5

    except Exception as error:
        print(f"Recommendation error: {error}")
    finally:
        curr.close()


def read_csv_data():
    movies = pd.read_csv('./data/movies.csv')
    ratings = pd.read_csv('./data/ratings.csv')
    return movies, ratings


if __name__ == '__main__':
    main()