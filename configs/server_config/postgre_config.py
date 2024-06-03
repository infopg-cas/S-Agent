from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy_utils import database_exists, create_database

POSTGRES_SIGNATURE_1 = 's-agent'
POSTGRES_SETTINGS = {
    POSTGRES_SIGNATURE_1: {
        "HOST": "localhost",
        "PORT": 5432,
        "USERNAME": "postgres",
        "PASSWORD": "9d36SkmzYV3#dssblr34b",
        "DB": "core",
        "CONNECTOR": "psycopg2"
    }
}


#  ----------------------- System functions ------------------------------ #
def get_db_uri(CONNECTOR, USERNAME, PASSWORD, HOST, PORT, DB):
    return f'postgresql+{CONNECTOR}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'


def get_db_session(db) -> Session:
    return sessionmaker(ENGINES[db])()


def auto_create_db(engine_name=POSTGRES_SIGNATURE_1):
    if not database_exists(ENGINES[engine_name].url):
        create_database(ENGINES[engine_name].url)
        print(f"Database Created: {database_exists(ENGINES[engine_name].url)}")


#  ----------------------- System functions ------------------------------ #

# Engine AND BASE setting, can add as many as wanted
DB_URI_1 = get_db_uri(**POSTGRES_SETTINGS[POSTGRES_SIGNATURE_1])

ENGINES = {
    POSTGRES_SIGNATURE_1: create_engine(DB_URI_1, max_overflow=-1)
}

auto_create_db()

BASES = {
    POSTGRES_SIGNATURE_1: declarative_base(ENGINES[POSTGRES_SIGNATURE_1])
}
